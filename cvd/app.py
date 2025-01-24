import clang.cindex


import streamlit as st
import torch
from torch_geometric.data import DataLoader, Data
import clang.cindex
import numpy as np
from sklearn.metrics import accuracy_score

# Import your GCN model and dataset processing methods
from gcn_def import GCN, clang_process
import os

os.environ['LIBCLANG_PATH'] = '/usr/lib/llvm-12/lib'
clang.cindex.Config.set_library_file('/usr/lib/llvm-12/lib/libclang.so')


# Set up the Streamlit app
st.title("Code Vulnerability Detector")
st.write("""
This application uses a GCN-based model to detect vulnerabilities in source code. 
Upload a code file to analyze its potential security risks.
""")

# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GCN().to(device)
model_path = "gnn_model.pth"  # Path to your saved model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# File upload
uploaded_file = st.file_uploader("Upload your source code file", type=["c", "cpp"])
if uploaded_file is not None:
    # Read the uploaded file
    code = uploaded_file.read().decode("utf-8")
    filename = uploaded_file.name
    st.code(code, language="c")

    # Preprocess the code to extract graph data
    st.write("Processing the code for analysis...")
    try:
        class TestCase:
            def __init__(self, filename, code):
                self.filename = filename
                self.code = code
                self.bug = 0  # Placeholder label

        testcase = TestCase(filename, code)
        data = clang_process(testcase)

        # Perform inference
        data = data.to(device)
        prediction = model(data.x.float(), data.edge_index, torch.tensor([0], device=device))
        prediction = torch.sigmoid(prediction).item()

        # Display result
        st.subheader("Prediction")
        if prediction >= 0.5:
            st.error(f"The code is likely vulnerable (Score: {prediction:.2f})")
        else:
            st.success(f"The code is likely safe (Score: {prediction:.2f})")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# Example Section
st.sidebar.header("Examples")
st.sidebar.write("Download sample code files to test the app:")
st.sidebar.download_button(
    label="Download Sample Vulnerable Code",
    data="int main() { int *ptr = NULL; *ptr = 42; return 0; }",
    file_name="vulnerable_code.c"
)
st.sidebar.download_button(
    label="Download Sample Safe Code",
    data="int main() { int x = 42; printf('%d', x); return 0; }",
    file_name="safe_code.c"
)

# Display metrics section
if st.button("Show Model Summary"):
    st.subheader("Model Architecture")
    st.text(str(model))
    st.write(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

st.write("Developed with ❤️ using Streamlit and PyTorch Geometric.")

