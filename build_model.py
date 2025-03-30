# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization, CategoryEncoding
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
import warnings
import joblib
import shap
from IPython.display import display
from sklearn.metrics import mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# %% [markdown]
# ## Data preparation

# %%
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('Y', axis=1)
y_train = train_df['Y']
X_test = test_df.drop('Y', axis=1)
y_test = test_df['Y']

X_train.shape[1]
X_train.head()

# %%
print(X_train['X10'].nunique())

# %%
# TODO: automatically infer from column dtype

numeric_cols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'] 
categorical_cols = ['X10']


normalizer = Normalization()
normalizer.adapt(np.array(X_train[numeric_cols])) 

categories = X_train['X10'].unique()
cat_mapping = {cat: i for i, cat in enumerate(categories)}

X_train['X10_encoded'] = X_train['X10'].map(cat_mapping)
X_test['X10_encoded'] = X_test['X10'].map(cat_mapping)

# Save the mapping for inference
joblib.dump(cat_mapping, 'categorical_mapping.joblib')

one_hot_encoder = CategoryEncoding(num_tokens=len(cat_mapping), output_mode="one_hot")

numeric_inputs = tf.keras.Input(shape=(len(numeric_cols),), name="numeric_inputs")
categorical_inputs = tf.keras.Input(shape=(1,), name="categorical_inputs")

numeric_preprocessed = normalizer(numeric_inputs)
categorical_preprocessed = one_hot_encoder(categorical_inputs)

concatenated_inputs = tf.keras.layers.Concatenate()([numeric_preprocessed, categorical_preprocessed])


# %%
# Note: we can't use sequential() as this only works when layers are stacked in a single flow, but we are concatenating the inputs

# x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(concatenated_inputs)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(concatenated_inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
output = tf.keras.layers.Dense(1, kernel_regularizer=regularizers.l2(1e-4))(x)

model = tf.keras.Model(inputs=[numeric_inputs, categorical_inputs], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_absolute_error')

model.summary()
    

# %%
history = model.fit(
    {
        "numeric_inputs": X_train[numeric_cols].values,
        "categorical_inputs": X_train['X10_encoded'].values.reshape(-1, 1)
    },
    y_train,
    validation_data=(
        {
            "numeric_inputs": X_test[numeric_cols].values,
            "categorical_inputs": X_test['X10_encoded'].values.reshape(-1, 1)
        },
        y_test
    ),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
)


model.save('dius_model.keras')


# %%
plt.figure(figsize=(12, 5))
    
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Test Loss', color='red', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

plt.show()

# %% [markdown]
# ## Feature Significance
# We use the `shap` values to compute feature significance

# %%

# wrapper function for the model that takes a single input
def f(x):
    return model({
        "numeric_inputs": x[:, :len(numeric_cols)],
        "categorical_inputs": x[:, len(numeric_cols):].reshape(-1, 1)
    })

# subset for efficiency
background_data = np.hstack([
    X_train[numeric_cols].values[:1000],
    X_train['X10_encoded'].values.reshape(-1, 1)[:1000]
])

explainer = shap.KernelExplainer(f, background_data)

# subset the test data for efficiency
test_data = np.hstack([
    X_test[numeric_cols].values[:200],
    X_test['X10_encoded'].values.reshape(-1, 1)[:200]
])
shap_values = explainer.shap_values(test_data)


if isinstance(shap_values, list):
    shap_values = shap_values[0]

feature_names = numeric_cols + ['X10']

feature_importance = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance.flatten() if feature_importance.ndim > 1 else feature_importance
})
importance_df = importance_df.sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
display(importance_df)

importance_df = importance_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance (Mean |SHAP Value|)', fontsize=14, fontweight='bold')
plt.xlabel('Mean |SHAP Value|', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()



# %% [markdown]
# ## Model Evaluation and Inference

# %%

# Check for any values in X_test['X10'] that are not in the mapping
unknown_categories = X_test['X10'].isin(cat_mapping.keys()) == False
if unknown_categories.any():
    print(f"Warning: Found {unknown_categories.sum()} unknown categories in test data.")
    print(f"Unknown categories: {X_test.loc[unknown_categories, 'X10'].unique()}")
    default_category = next(iter(cat_mapping.values()))
    X_test['X10_encoded_safe'] = X_test['X10'].apply(lambda x: cat_mapping.get(x, default_category))
else:
    X_test['X10_encoded_safe'] = X_test['X10_encoded']

y_pred = model.predict({
    "numeric_inputs": X_test[numeric_cols].values,
    "categorical_inputs": X_test['X10_encoded_safe'].values.reshape(-1, 1)
})

test_df['prediction'] = y_pred
test_df.to_csv('test_pred.csv')

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) on test data: {mae:.4f}")

# %%
test_df[['Y', 'prediction']].head(10)

# %% [markdown]
# 


