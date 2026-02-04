import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# ================= CONFIG =================
TRAIN = True
MODEL_PATH = "mentor_model.keras"
# ========================================

print("Starting...")

# ================= LOAD =================

train_df = pd.read_excel("train.xlsx")
test_df  = pd.read_excel("test.xlsx")
targets  = pd.read_csv("target.csv")

print("Loaded files")

# ================= PREPROCESS =================

num_cols = ["Age","Company_Size_Employees"]
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

train_text = train_df[text_cols].fillna("").agg(" ".join,axis=1)
test_text  = test_df[text_cols].fillna("").agg(" ".join,axis=1)

tfidf = TfidfVectorizer(max_features=500)
train_tfidf = tfidf.fit_transform(train_text).toarray()
test_tfidf  = tfidf.transform(test_text).toarray()

X_train = np.hstack([train_num,train_cat,train_tfidf])
X_test  = np.hstack([test_num,test_cat,test_tfidf])

print("Profiles built")

# ================= BUILD POSITIVE PAIRS =================

id_map = {pid:i for i,pid in enumerate(train_df["Profile_ID"])}

XA=[]
XB=[]
y=[]

for _,r in targets.iterrows():
    XA.append(X_train[id_map[r.src_user_id]])
    XB.append(X_train[id_map[r.dst_user_id]])
    y.append(r.compatibility_score)

XA = np.array(XA)
XB = np.array(XB)
y  = np.array(y)

# mirror
XA0,XB0,y0 = XA.copy(),XB.copy(),y.copy()
XA = np.concatenate([XA0,XB0])
XB = np.concatenate([XB0,XA0])
y  = np.concatenate([y0,y0])

# ================= NEGATIVE SAMPLING (FAST) =================

num_pos = len(y0)
idx = np.random.choice(len(X_train),(num_pos,2),replace=True)

neg_XA = X_train[idx[:,0]]
neg_XB = X_train[idx[:,1]]
neg_y  = np.zeros(num_pos)

XA = np.concatenate([XA,neg_XA])
XB = np.concatenate([XB,neg_XB])
y  = np.concatenate([y,neg_y])

# normalize + shuffle
y = (y-y.min())/(y.max()-y.min()+1e-8)

p = np.random.permutation(len(y))
XA,XB,y = XA[p],XB[p],y[p]

print("Training samples:",len(y))

# ================= MODEL =================

D = X_train.shape[1]

userNN = tf.keras.Sequential([
    tf.keras.layers.Dense(256,activation="relu",input_shape=(D,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(64)
])

A = tf.keras.Input(shape=(D,))
B = tf.keras.Input(shape=(D,))

VA = tf.keras.layers.LayerNormalization()(userNN(A))
VB = tf.keras.layers.LayerNormalization()(userNN(B))

diff = tf.keras.layers.Subtract()([VA,VB])
diff = tf.keras.layers.Lambda(lambda x: tf.abs(x))(diff)

dot  = tf.keras.layers.Dot(axes=1)([VA,VB])

merged = tf.keras.layers.Concatenate()([VA,VB,diff,dot])

x = tf.keras.layers.Dense(128,activation="relu")(merged)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(1,activation="sigmoid")(x)

model = tf.keras.Model([A,B],out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy"
)

# ================= TRAIN / LOAD =================

if TRAIN or not os.path.exists(MODEL_PATH):
    print("Training...")
    model.fit([XA,XB],y,batch_size=64,epochs=40,validation_split=0.2)
    model.save(MODEL_PATH)
    print("Model saved")
else:
    model = tf.keras.models.load_model(MODEL_PATH,safe_mode=False)

# ================= FULL TEST CARTESIAN =================

ids = test_df["Profile_ID"].values
N=len(X_test)

print("Generating",N*N,"pairs")

A_test = np.repeat(X_test,N,axis=0)
B_test = np.tile(X_test,(N,1))

pairs = np.repeat(ids,N).astype(str)+"_"+np.tile(ids,N).astype(str)

pred = model.predict([A_test,B_test],batch_size=512).flatten()

pd.DataFrame({
    "ID":pairs,
    "compatibility_score":pred
}).to_csv("submission.csv",index=False)

print("Saved submission.csv")
print("Prediction range:",pred.min(),pred.max())
