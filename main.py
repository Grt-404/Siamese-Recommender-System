import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


train_df = pd.read_excel("train.xlsx")
test_df  = pd.read_excel("test.xlsx")
targets  = pd.read_csv("target.csv")
print("Loaded data files.")



num_cols = ["Age", "Company_Size_Employees"]
cat_cols = ["Gender","Role","Seniority_Level","Industry","Location_City"]
text_cols = ["Business_Interests","Business_Objectives","Constraints"]

train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].mean())
test_df[num_cols]  = test_df[num_cols].fillna(train_df[num_cols].mean())

scaler = StandardScaler()
train_num = scaler.fit_transform(train_df[num_cols])
test_num  = scaler.transform(test_df[num_cols])

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_cat = ohe.fit_transform(train_df[cat_cols].fillna("missing"))
test_cat  = ohe.transform(test_df[cat_cols].fillna("missing"))

train_text = train_df[text_cols].fillna("").agg(" ".join, axis=1).str.replace(";"," ")
test_text  = test_df[text_cols].fillna("").agg(" ".join, axis=1).str.replace(";"," ")

tfidf = TfidfVectorizer(max_features=500)
train_tfidf = tfidf.fit_transform(train_text).toarray()
test_tfidf  = tfidf.transform(test_text).toarray()

X_train_profiles = np.hstack([train_num, train_cat, train_tfidf])
X_test_profiles  = np.hstack([test_num, test_cat, test_tfidf])
print("Finished preprocessing profiles.")


id_to_index = {pid:i for i,pid in enumerate(train_df["Profile_ID"])}

XA, XB, y = [], [], []

for _,row in targets.iterrows():
    i = id_to_index[row["src_user_id"]]
    j = id_to_index[row["dst_user_id"]]
    XA.append(X_train_profiles[i])
    XB.append(X_train_profiles[j])
    y.append(row["compatibility_score"])

XA = np.array(XA)
XB = np.array(XB)
y  = np.array(y)
print("Built labeled pairs.")


XA_orig = XA.copy()
XB_orig = XB.copy()
y_orig  = y.copy()

XA = np.concatenate([XA_orig, XB_orig])
XB = np.concatenate([XB_orig, XA_orig])
y  = np.concatenate([y_orig,  y_orig])


num_pos = len(y_orig)
num_profiles = X_train_profiles.shape[0]

neg_XA, neg_XB, neg_y = [], [], []
neg_pairs = set()

while len(neg_y) < num_pos:
    i, j = np.random.choice(num_profiles, 2, replace=False)
    neg_XA.append(X_train_profiles[i])
    neg_XB.append(X_train_profiles[j])
    neg_y.append(0.0)

XA = np.concatenate([XA, np.array(neg_XA)])
XB = np.concatenate([XB, np.array(neg_XB)])
y  = np.concatenate([y,  np.array(neg_y)])
print("Finished negative sampling and mirroring pairs.")

# Normalize targets
y = (y - y.min()) / (y.max() - y.min() + 1e-8)

# Shuffle
perm = np.random.permutation(len(XA))
XA = XA[perm]
XB = XB[perm]
y  = y[perm]

print("y range:", y.min(), y.max())
print("Ready to build model.")


D = X_train_profiles.shape[1]

user_NN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64)
])

second_NN = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(192,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

input_a = tf.keras.Input(shape=(D,))
input_b = tf.keras.Input(shape=(D,))

VA = user_NN(input_a)
VB = user_NN(input_b)

VA = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(VA)
VB = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(VB)

diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0]-x[1]))([VA,VB])
merged = tf.keras.layers.Concatenate(axis=1)([VA,VB,diff])

output = second_NN(merged)

model = tf.keras.Model([input_a,input_b], output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.MeanSquaredError()
)

model.summary()


cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
print("Starting training...")
model.fit(
    [XA,XB],
    y,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=[cb]
)


pair_ids = []
test_XA = []
test_XB = []

test_ids = test_df["Profile_ID"].values

N = min(200, len(X_test_profiles))

for i in range(N):
    for j in range(N):
        if i != j:
            test_XA.append(X_test_profiles[i])
            test_XB.append(X_test_profiles[j])
            pair_ids.append(str(test_ids[i]) + "_" + str(test_ids[j]))

test_XA = np.array(test_XA)
test_XB = np.array(test_XB)

preds = model.predict([test_XA,test_XB]).flatten()

out = pd.DataFrame({
    "pair": pair_ids,
    "compatibility_score": preds
})

out.to_csv("submission.csv", index=False)

print("Saved submission.csv")
print("Prediction range:", preds.min(), preds.max())
