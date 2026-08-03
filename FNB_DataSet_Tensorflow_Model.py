##Please note this requires tensorflow 2.12 which will then come wiht keras 2.12. It does not work with Keras>3
print("hello FNB")
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
from collections import defaultdict
masterdf = pd.read_csv('data_2.csv')
masterdf = masterdf[~masterdf["interaction"].str.contains("DISPLAY", case=False, na=False)]
masterdf = masterdf.sort_values(by="idcol")
test_set = masterdf.iloc[:499].copy()
masterdf = masterdf.iloc[499:].reset_index(drop=True)

##new user Test Set Setup
test_set_unique_interaction =test_set.drop_duplicates(subset=['interaction', 'idcol', "item"])
mask = (test_set_unique_interaction["interaction"] == "CLICK") & \
       (test_set_unique_interaction["interaction"].shift(-1) == "CHECKOUT")
rows_to_drop = test_set_unique_interaction[mask].index
test_set_unique_interaction = test_set_unique_interaction.drop(rows_to_drop)
test_set_unique_interaction = test_set_unique_interaction.reset_index(drop=True)
##end
##Setting up the data
test_set_unique = test_set.drop_duplicates(subset='idcol', keep='first')#Used for the recommendations
user_df2 = test_set_unique_interaction[["item", "segment", "beh_segment", "active_ind","idcol"]]

user_df= masterdf[["item", "segment", "beh_segment", "active_ind","idcol"]]
item_df = masterdf[["item"]].drop_duplicates()
user_ds = tf.data.Dataset.from_tensor_slices({
    "idcol": tf.convert_to_tensor(user_df["idcol"].astype(int)),
    "item": tf.convert_to_tensor(user_df["item"].astype(str)),
    "beh_segment": tf.convert_to_tensor(user_df["beh_segment"].astype(str)),
    "active_ind": tf.convert_to_tensor(user_df["active_ind"].astype(str)),
    "segment": tf.convert_to_tensor(user_df["segment"].astype(str)),
})
item_ds = tf.data.Dataset.from_tensor_slices({
    "item": tf.convert_to_tensor(item_df["item"].astype(str)),})

unique_user_ids = user_df["idcol"].astype(int).unique()
unique_segment = user_df["segment"].astype(str).unique()
unique_Items = item_df["item"].astype(str).unique()
unique_beh_segment = user_df["beh_segment"].astype(str).unique()
unique_active_ind = user_df["active_ind"].astype(str).unique()
##end

##building model
class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_ids_embedding = tf.keras.Sequential([tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None), tf.keras.layers.Embedding(len(unique_user_ids)+1, 32)])
        self.user_beh_segment_embedding = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_beh_segment, mask_token=None), tf.keras.layers.Embedding(len(unique_beh_segment)+1, 32)])
        self.user_segment = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_segment, mask_token=None), tf.keras.layers.Embedding(len(unique_segment)+1, 32)])
        self.user_active_ind = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_active_ind, mask_token=None), tf.keras.layers.Embedding(len(unique_active_ind)+1, 32)])

    def call(self, inputs):
        return tf.concat([self.user_ids_embedding(inputs["idcol"]), self.user_beh_segment_embedding(inputs["beh_segment"]),self.user_segment(inputs["segment"]), self.user_active_ind(inputs["active_ind"]),], axis=1)

class QueryModel(tf.keras.Model):
    def __init__(self, layer_sizes, projection_dim=None):
        super().__init__()
        #defining the user model
        self.embedding_model = UserModel()
        self.dense_layers = tf.keras.Sequential()
        ##ading the layers up until the last one, which doesn't require an activation function
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
        ##no activation function
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        #First line is processing the inputs and returning the vector embeddings
        feature_embedding = self.embedding_model(inputs)
        ## this returns the output of the neural network after layers to capture different patterns in data
        return self.dense_layers(feature_embedding)

class ItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.Items_embedding = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_Items,mask_token=None),tf.keras.layers.Embedding(len(unique_Items) + 1, 32)])

    def call(self, inputs):
        return tf.concat([self.Items_embedding(inputs["item"]),], axis=1)

class CandidateModel(tf.keras.Model):
    def __init__(self, layer_sizes):
        super().__init__()
        self.embedding_model = ItemModel()
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)

class BankingModel(tfrs.models.Model):
    def __init__(self, layer_sizes):
        super().__init__()
        ##Creates our models and puts in the neural network sizes. Layer sizes are arrays I think
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=item_ds.batch(128).map(lambda x: (x["item"], self.candidate_model(x)))))
    def compute_loss(self, features, training=False):
        print(features.keys())
        query_embeddings = self.query_model({"idcol": features["idcol"],"beh_segment": features["beh_segment"], "segment": features["segment"], "active_ind": features["active_ind"],})
        movie_embeddings = self.candidate_model({"item": features["item"],})
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)
##end

##training model
tf.random.set_seed(42)
shuffled = user_ds.shuffle(220_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(200_000)
validate = shuffled.skip(200_000).take(15_000)
test= shuffled.skip(215_000).take(5000)
cached_train = train.shuffle(200_000).batch(2048)
cached_validate = validate.batch(4096).cache()
cached_test = test.batch(500).cache()
num_epochs = 5
model = BankingModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
one_layer_history = model.fit(cached_train, validation_data=cached_validate, validation_freq=5, epochs=num_epochs, verbose=0)
accuracy = one_layer_history.history["val_factorized_top_k/top_10_categorical_accuracy"][-1]
print(f"Top-10 accuracy: {accuracy:.2f}.")
##end

##test set but from users that the model has already seen
test_results = model.evaluate(cached_test, return_dict=True)
# Extract the top-10 accuracy (adjust key name based on your metrics)
accuracy = test_results.get("factorized_top_k/top_10_categorical_accuracy", None)
print(accuracy)
##end

#test set of completely unseen data(i.e. unseen users)
user_ds3 = tf.data.Dataset.from_tensor_slices({
    "idcol": tf.convert_to_tensor(test_set["idcol"].astype(int)),
    "item": tf.convert_to_tensor(test_set["item"].astype(str)),
    "beh_segment": tf.convert_to_tensor(test_set["beh_segment"].astype(str)),
    "active_ind": tf.convert_to_tensor(test_set["active_ind"].astype(str)),
    "segment": tf.convert_to_tensor(test_set["segment"].astype(str)),
})
user_ds3 = user_ds3.batch(128)
test_results = model.evaluate(user_ds3, return_dict=True)
accuracy = test_results.get("factorized_top_k/top_10_categorical_accuracy", None)
print(accuracy)
##end


##recommending
print("\n--- Generating Predictions ---")
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
print("Indexing candidate items...")
index.index_from_dataset(
    item_ds.batch(128).map(lambda x: (x["item"], model.candidate_model(x)))
)
print("Candidate indexing complete.")

##precision calculation

def precision_at_k(model, test_dataset, index, k=10):
    user_interactions = defaultdict(set)
    user_features_lookup = {}

    for example in test_dataset:
        user_id = int(example["idcol"].numpy())
        item_id = example["item"].numpy().decode("utf-8")
        user_interactions[user_id].add(item_id)

        if user_id not in user_features_lookup:
            user_features_lookup[user_id] = {
                "idcol": tf.expand_dims(example["idcol"], 0),
                "beh_segment": tf.expand_dims(example["beh_segment"], 0),
                "segment": tf.expand_dims(example["segment"], 0),
                "active_ind": tf.expand_dims(example["active_ind"], 0),
            }

    precisions = []

    for user_id, true_items in user_interactions.items():
        user_features = user_features_lookup[user_id]
        _, recommended_items = index(user_features, k=k)
        recommended_items_list = [item.decode("utf-8") for item in recommended_items[0].numpy()]
        hits = len(set(recommended_items_list) & true_items)
        precision = hits / k
        precisions.append(precision)

    return np.mean(precisions)

mean_precision = precision_at_k(model, test, index, k=10)
print(mean_precision)

def recall_at_k(model, test_ds, index, k=10):
    total = 0
    hits = 0

    for example in test_ds:
        input_dict = {
            "idcol": tf.expand_dims(example["idcol"], axis=0),
            "beh_segment": tf.expand_dims(example["beh_segment"], axis=0),
            "segment": tf.expand_dims(example["segment"], axis=0),
            "active_ind": tf.expand_dims(example["active_ind"], axis=0),
        }
        _, recommended_items = index(input_dict, k=k)
        true_item = example["item"].numpy()
        if isinstance(true_item, bytes):
            true_item = true_item.decode("utf-8")

        recommended_items = [x.numpy().decode("utf-8") if isinstance(x.numpy(), bytes) else x.numpy()
                             for x in recommended_items[0]]

        if true_item in recommended_items:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


recall = recall_at_k(model, test, index, 10)
print(recall)

def compute_item_popularity(test_dataset):
    item_counts = defaultdict(int)
    total_interactions = 0

    for example in test_dataset:
        item_id = example["item"].numpy().decode("utf-8")
        item_counts[item_id] += 1
        total_interactions += 1
    item_popularity = {item: count / total_interactions for item, count in item_counts.items()}
    return item_popularity


def novelty_at_k(model, test_dataset, index, k=10):
    item_popularity = compute_item_popularity(test_dataset)
    user_interactions = defaultdict(set)
    user_features_lookup = {}

    for example in test_dataset:
        user_id = int(example["idcol"].numpy())
        item_id = example["item"].numpy().decode("utf-8")
        user_interactions[user_id].add(item_id)

        if user_id not in user_features_lookup:
            user_features_lookup[user_id] = {
                "idcol": tf.expand_dims(example["idcol"], 0),
                "beh_segment": tf.expand_dims(example["beh_segment"], 0),
                "segment": tf.expand_dims(example["segment"], 0),
                "active_ind": tf.expand_dims(example["active_ind"], 0),
            }

    novelties = []

    for user_id in user_interactions:
        user_features = user_features_lookup[user_id]
        _, recommended_items = index(user_features, k=k)
        recommended_items_list = [item.decode("utf-8") for item in recommended_items[0].numpy()]
        epsilon = 1e-10
        item_novelties = [
            -np.log2(item_popularity.get(item, epsilon)) for item in recommended_items_list
        ]
        avg_novelty = np.mean(item_novelties)
        novelties.append(avg_novelty)

    mean_novelty = np.mean(novelties)
    return mean_novelty
novelty_score = novelty_at_k(model, test, index, k=10)
print(novelty_score)
