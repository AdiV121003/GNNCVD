import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from io import StringIO

# Define the GCN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)  # Sigmoid activation for binary classification

# Load the trained model
@st.cache_resource
def load_model():
    model = GCN(input_dim=16, hidden_dim=32, output_dim=1)  # Adjust input_dim as needed
    model.load_state_dict(torch.load("gcn_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title("🔍 Code Vulnerability Detector")

# Upload CSV File
uploaded_file = st.file_uploader("Upload a CSV file with code snippets", type=["csv"])

if uploaded_file is not None:
    # Read CSV File
    df = pd.read_csv(uploaded_file)

    # Display uploaded file
    st.write("📂 Uploaded Data:")
    st.dataframe(df.head())

    # Preprocess data
    def preprocess_data(df):
        data_list = []
        for _, row in df.iterrows():
            # Convert 'code' into a numerical representation (dummy example)
            x = torch.randn((10, 16))  # Example feature tensor (adjust as needed)
            edge_index = torch.randint(0, 10, (2, 20))  # Example graph structure
            y = torch.tensor([1.0]) if row["bug"] else torch.tensor([0.0])  # Binary labels
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list

    # Convert dataframe to graph data
    graph_data = preprocess_data(df)

    # Make predictions
    predictions = []
    for data in graph_data:
        with torch.no_grad():
            pred = model(data).item()
            predictions.append(pred > 0.5)  # Thresholding at 0.5

    # Add predictions to dataframe
    df["Predicted Vulnerability"] = ["✅ Safe" if not p else "⚠️ Vulnerable" for p in predictions]

    # Show results
    st.subheader("🔎 Results")
    st.dataframe(df)

    # Download results
    csv_output = StringIO()
    df.to_csv(csv_output, index=False)
    st.download_button("📥 Download Results", csv_output.getvalue(), "results.csv", "text/csv")

st.write("Developed with ❤️ using Streamlit & PyTorch Geometric")
