import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn

class LinearRegressionModel:
    def __init__(self, df):
        self.df = df
        # Extracting features and label from the dataframe
        self.features = self.df[['Years_of_Experience']]
        self.label = self.df['Salary']

        # Hyperparameters
        self.learning_rate = 0.00001
        self.batch_size = 2
        self.epochs = 50

        self.model_output = None

        # Splitting data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.label, test_size=0.2, random_state=42
        )

    def build_model(self):
        model = keras.Sequential()
        # Fixing input shape for single feature
        model.add(keras.layers.Dense(units=1, input_shape=(1,)))  # Input shape should be (1,)
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                      loss='mean_squared_error',
                      metrics=[keras.metrics.RootMeanSquaredError()])
        self.model = model

    def train_model(self):
        history = self.model.fit(x=self.x_train, y=self.y_train, epochs=self.epochs, batch_size=self.batch_size)
        weights, bias = self.model.get_weights()
        epochs = history.epoch
        rmse = pd.DataFrame(history.history)["root_mean_squared_error"]
        self.model_output = (weights, bias, epochs, rmse)

    def plot_loss_curve(self, fig):
        _, _, epochs, rmse = self.model_output
        curve = px.line(x=epochs, y=rmse)
        curve.update_traces(line_color='#ff0000', line_width=3)
        fig.append_trace(curve.data[0], row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min() * 0.8, rmse.max()])

    def plot_model(self, fig):
        weights, bias, _, _ = self.model_output
        self.df['Predicted_Salary'] = bias[0]
        self.df['Predicted_Salary'] += weights[0][0] * self.df['Years_of_Experience']

        # Plotting the model line
        model_line = px.line(self.df, x='Years_of_Experience', y='Predicted_Salary')
        fig.add_trace(model_line.data[0], row=1, col=2)

    def make_plots(self):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss Curve", "Model Plot"),
                            specs=[[{"type": "scatter"}, {"type": "scatter"}]])
        self.plot_loss_curve(fig)
        self.plot_model(fig)
        fig.show()

    def predict(self, batch_size=5):
      # Ensure the batch size is not larger than the dataset
      batch_size = min(batch_size, len(self.df))
      
      # Sample the batch from the dataset
      batch = self.df.sample(n=batch_size).copy()
      batch.reset_index(drop=True, inplace=True)
      
      # Use the feature column name correctly
      predictions = self.model.predict_on_batch(batch[['Years_of_Experience']])

      output = pd.DataFrame({
          "PREDICTED_SALARY": predictions.flatten(),
          "OBSERVED_SALARY": batch['Salary'],
          "L1_LOSS": abs(predictions.flatten() - batch['Salary'])
      })

      return output


    def run_experiment(self):
        self.build_model()
        self.train_model()
        self.make_plots()
        return self.model_output


# Load your dataset
dataset = pd.read_csv("/content/tableConvert.com_b10cel (2).csv")

# Initialize the LinearRegressionModel class
model = LinearRegressionModel(dataset)

# Run the experiment to train the model and visualize the results
model.run_experiment()

# Make predictions and display the output
output = model.predict()
print(output.head())
