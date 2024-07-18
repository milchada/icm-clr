from scripts.data.CustomDataset import CustomDataset
import optuna, os, yaml
import config as c 

params = yaml.safe_load(open('params.yaml'))
train_default_params = params['train_mlp']
df_path = c.dataset_raw_path + params['extract']['DATASETS'][0] + '/label.csv'
pred_labels = params['prepare']['UNOBSERVABLES'][0]

# Define hyperparameters
learning_rate = train_default_params['LEARNING_RATE']
num_epochs = train_default_params['NUM_EPOCHS']
batch_size = train_default_params['BATCH_SIZE']
image_size = train_default_params['IMAGE_SIZE']
ndim_rep = params['model']['RESNET_REPRESENTATION_DIM']

from keras.layers import Conv2D, Activation, MaxPool2D, Input, Dense
from keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2))(x)
    return x

def build_cnn(input_shape, output_channels=len(pred_labels), activation="relu"):
    inputs = Input(shape=input_shape, name='image_input')
    
    x = conv_block(inputs, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_channels, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def train(df_path, pred_labels, transform=None, save_path=c.model_path+'/mlp.pt'):
    custom_dataset = CustomDataset(csv_file=df_path, pred_labels = pred_labels, transform=transform)
    train_size = int(len(custom_dataset)*0.9)
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, len(custom_dataset) - train_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_cnn((image_size, image_size, 1)) 
    model.compile(optimizer='adam', loss='mse')

    history = model.fit(x = train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(filepath=save_path,
                monitor='val_loss',
                patience=5,
                restore_best_weights=True)])


def load_model(save_path):
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

if __name__ == "__main__":
    train(df_path, pred_labels, save_path='fiducial/model/mlp.pt')