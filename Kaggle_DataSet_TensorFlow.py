##Please note this requires tensorflow 2.12 which will then come wiht keras 2.12. It does not work with Keras>3
print("Hello FNB")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from collections import defaultdict
##Reading in and removing the NA's
masterdf = pd.read_csv('Commerce.csv', encoding='ISO-8859-1')
print(len(masterdf))
masterdf = masterdf.dropna(subset=["CustomerID", "UnitPrice", "Description", "Country", "StockCode"])

user_interaction_counts = masterdf["CustomerID"].value_counts()
active_users = user_interaction_counts[user_interaction_counts >= 3].index
filtered_df = masterdf[masterdf["CustomerID"].isin(active_users)]
filtered_df = filtered_df.reset_index(drop=True)
print(filtered_df)


user_df = filtered_df[["CustomerID", "Description", "Country", "StockCode"]]
item_df = filtered_df[["StockCode", "Description"]].drop_duplicates()
##Creating the tensors
user_ds = tf.data.Dataset.from_tensor_slices({
    "CustomerID": tf.convert_to_tensor(user_df["CustomerID"].astype(int)),
    "Description": tf.convert_to_tensor(user_df["Description"].astype(str)),
    "Country": tf.convert_to_tensor(user_df["Country"].astype(str)),
    "StockCode": tf.convert_to_tensor(user_df["StockCode"].astype(str)),
})
item_ds = tf.data.Dataset.from_tensor_slices({
    "StockCode": tf.convert_to_tensor(item_df["StockCode"].astype(str)),
    "Description": tf.convert_to_tensor(item_df["Description"].astype(str)),
})
##end

##Creating the unique identifiers
unique_user_ids = user_df["CustomerID"].astype(int).unique()
unique_countries = user_df["Country"].astype(str).unique()
unique_Items = item_df["StockCode"].astype(str).unique()
unique_description = item_df["Description"].astype(str).unique()
##end

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_ids_embedding = tf.keras.Sequential([tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None), tf.keras.layers.Embedding(len(unique_user_ids)+1, 32)])
        self.user_country_embedding = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_countries, mask_token=None), tf.keras.layers.Embedding(len(unique_countries)+1, 32)])

    def call(self, inputs):
        return tf.concat([self.user_ids_embedding(inputs["CustomerID"]), self.user_country_embedding(inputs["Country"])], axis=1)

class QueryModel(tf.keras.Model):
    def __init__(self, layer_sizes, projection_dim=None):
        super().__init__()
        self.embedding_model = UserModel()
        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)

class ItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        max_tokens = 10_000
        self.title_embedding = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_Items,mask_token=None),tf.keras.layers.Embedding(len(unique_Items) + 1, 32)])
        max_tokens = 10_000
        self.description_embedding = tf.keras.Sequential([tf.keras.layers.StringLookup(vocabulary=unique_description,mask_token=None),tf.keras.layers.Embedding(len(unique_description) + 1, 32)])
    def call(self, inputs):
       return tf.concat([self.title_embedding(inputs["StockCode"]), self.description_embedding(inputs["Description"])], axis=1)

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

class ShoppingModel(tfrs.models.Model):
    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(candidates=item_ds.batch(128).map(lambda x: (x["StockCode"],self.candidate_model({"StockCode": x["StockCode"],"Description": x["Description"],})))))
    def compute_loss(self, features, training=False):
        print(features.keys())
        query_embeddings = self.query_model({"CustomerID": features["CustomerID"], "Country": features["Country"]})
        movie_embeddings = self.candidate_model({"StockCode": features["StockCode"], "Description": features["Description"]})
        return self.task(query_embeddings, movie_embeddings, compute_metrics=not training)

tf.random.set_seed(42)
shuffled = user_ds.shuffle(400_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(350_000)
validate = shuffled.skip(360_000).take(40_000)
test = shuffled.skip(390000).take(10000)
cached_train = train.shuffle(360_000).batch(2048)
cached_validate = validate.batch(4096).cache()
cached_test = test.batch(1000).cache()

num_epochs = 5
model = ShoppingModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
one_layer_history = model.fit(cached_train, validation_data=cached_test, validation_freq=5, epochs=num_epochs, verbose=0)
accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")


test_results = model.evaluate(cached_test, return_dict=True)
accuracy = test_results.get("factorized_top_k/top_100_categorical_accuracy", None)
print(accuracy)

print("\n--- Generating Predictions ---")
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
print("Indexing candidate items...")
index.index_from_dataset(item_ds.batch(128).map(lambda x: (x["StockCode"], model.candidate_model(x)))
)
print("Candidate indexing complete.")

def precision_at_k(model, test_ds, index, k=10):
    total_precision = 0
    num_users = 0

    for batch in test_ds:
        user_ids = batch["CustomerID"].numpy()
        true_items = batch["StockCode"].numpy()

        scores, recommendations = index({
            "CustomerID": batch["CustomerID"],
            "Country": batch["Country"]
        }, k=k)

        for i in range(len(user_ids)):
            recommended_items = recommendations[i].numpy().astype(str)
            actual_item = true_items[i].decode("utf-8") if isinstance(true_items[i], bytes) else str(true_items[i])

            if actual_item in recommended_items:
                total_precision += 1

            num_users += 1

    return total_precision / num_users if num_users > 0 else 0.0



def recall_at_k(model, test_ds, index, k=10):
    total_recall = 0
    num_users = 0

    for batch in test_ds:
        user_ids = batch["CustomerID"].numpy()
        true_items = batch["StockCode"].numpy()

        _, recommendations = index({
            "CustomerID": batch["CustomerID"],
            "Country": batch["Country"]
        }, k=k)

        for i in range(len(user_ids)):
            recommended_items = recommendations[i].numpy().astype(str)
            actual_item = true_items[i].decode("utf-8") if isinstance(true_items[i], bytes) else str(true_items[i])

            if actual_item in recommended_items:
                total_recall += 1

            num_users += 1

    return total_recall / num_users if num_users > 0 else 0.0

def get_item_popularity(dataset):
    popularity = defaultdict(int)
    for batch in dataset:
        items = batch["StockCode"].numpy()
        for i in items:
            if isinstance(i, bytes):
                item_id = i.decode("utf-8")
            else:
                item_id = str(i)
            popularity[item_id] += 1
    return popularity


item_popularity = get_item_popularity(test)

def novelty_at_k(model, test_ds, index, item_popularity, k=10):
    user_features_lookup = {}

    user_ids = set()
    user_countries = {}

    for batch in test_ds:
        for i, uid in enumerate(batch["CustomerID"].numpy()):
            user_ids.add(int(uid))
            user_countries[int(uid)] = batch["Country"].numpy()[i].decode("utf-8")

    for uid in user_ids:
        user_features_lookup[uid] = {
            "CustomerID": tf.expand_dims(tf.convert_to_tensor(uid), 0),
            "Country": tf.convert_to_tensor([user_countries[uid]])
        }

    novelties = []

    for user_id in user_ids:
        user_features = user_features_lookup[user_id]
        _, recommended_items = index(user_features, k=k)
        recommended_items_list = [item.decode("utf-8") for item in recommended_items[0].numpy()]

        novelty_scores = []
        for item in recommended_items_list:
            pop = item_popularity.get(item, 1)
            novelty_scores.append(1 / np.log(pop + 1))
        novelties.append(np.mean(novelty_scores))

    return np.mean(novelties)

p_at_50 = precision_at_k(model, cached_test, index,k=50)
r_at_50 = recall_at_k(model, cached_test, index,k=50)
n_at_50 = novelty_at_k(model, cached_test, index, item_popularity, 50)
print(p_at_50)
print(r_at_50)
print(n_at_50)




