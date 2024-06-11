import streamlit as st
import pickle
import numpy as np

# Load models
logr_model = pickle.load(open('LogR_Model.pkl', 'rb'))
forest_model = pickle.load(open('Forest_Model.pkl', 'rb'))
tree_model = pickle.load(open('Tree_Model.pkl', 'rb'))

# Classification function
def classify(num):
    if num == 0:
        return 'Không ngon'
    elif num == 1:
        return 'Ngon'
    else:
        return 'Unknown'

def main():
    st.title("Ramen Classification")
    
    # HTML template for header
    html_temp = """
    <div style="background-color:#006600; padding:10px">
    <h2 style="color:white;text-align:center;">Ramen Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Model selection
    activities = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    
    # List of all features
    features = [
    "IsSpicy", "HasChicken", "HasBeef", "HasSeafoods", "With_Bowl", "With_Cup", "With_Other",
    "With_Pack", "With_Tray", "from_Acecook", "from_Indomie", "from_KOKA", "from_Lucky Me!", "from_MAMA",
    "from_Maggi", "from_Mama", "from_Mamee", "from_Maruchan", "from_MyKuali", "from_Myojo", "from_Nissin",
    "from_Nongshim", "from_Other", "from_Ottogi", "from_Paldo", "from_Samyang Foods", "from_Sapporo Ichiban",
    "from_Sau Tao", "from_Ve Wong", "from_Vifon", "from_Vina Acecook", "from_Wai Wai", "In_China",
    "In_Hong Kong", "In_Indonesia", "In_Japan", "In_Malaysia", "In_Other", "In_Singapore", "In_South Korea",
    "In_Taiwan", "In_Thailand", "In_United States", "In_Vietnam"
    ]

    
    # Create input fields for each feature
    inputs = {}
    for feature in features:
        inputs[feature] = st.number_input(f"{feature}", min_value=0, max_value=1, value=0, key=feature)
    
    # Convert the input dictionary to a list of values
    input_values = list(inputs.values())

    # Reshape the list to a 2D array
    input_array = np.array(input_values).reshape(1, -1)
    
    # Classification
    if st.button('Submit'):
        if option == 'LogisticRegression':
            prediction = logr_model.predict(input_array)
        elif option == 'DecisionTreeClassifier':
            prediction = tree_model.predict(input_array)
        else:  # RandomForestClassifier
            prediction = forest_model.predict(input_array)
        
        st.write("Prediction:", prediction)
        
        result = classify(prediction[0])
        st.success(result)

if __name__ == '__main__':
    main()
