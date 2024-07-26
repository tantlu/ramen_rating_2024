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
        return 'Mì ramen không ngon'
    else:
        return 'Mì ramen ngon'

def main():
    st.title("Phân lớp Mì Ramen")
    
    # HTML template for header
    html_temp = """
    <div style="background-color:#006600; padding:10px">
    <h2 style="color:white;text-align:center;">Phân lớp Mì Ramen</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Model selection
    activities = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']
    option = st.sidebar.selectbox('Chọn Model mà bạn muốn sử dụng', activities)
    st.subheader(option)
    
    # Define feature groups
    base_features = ["IsSpicy", "HasChicken", "HasBeef", "HasSeafoods"]
    styles_features = ["With_Bowl", "With_Cup", "With_Other", "With_Pack", "With_Tray"]
    brand_features = ["from_Acecook", "from_Indomie", "from_KOKA", "from_Lucky Me!", "from_Maggi", 
                      "from_Mama", "from_Mamee", "from_Maruchan", "from_MyKuali", "from_Myojo", 
                      "from_Nissin", "from_Nongshim", "from_Other", "from_Ottogi", "from_Paldo", 
                      "from_Samyang Foods", "from_Sapporo Ichiban", "from_Sau Tao", "from_Ve Wong", 
                      "from_Vifon", "from_Vina Acecook"]
    country_features = ["In_China", "In_Hong Kong", "In_Indonesia", "In_Japan", "In_Malaysia", 
                       "In_Other", "In_Singapore", "In_South Korea", "In_Taiwan", "In_Thailand", 
                       "In_United States", "In_Vietnam"]

    # Create input fields for each feature group
    inputs = {}

    # Base features
    base_cols = st.columns(4)
    for idx, feature in enumerate(base_features):
        inputs[feature] = base_cols[idx % 4].radio(f"{feature}", options=[0, 1], index=0, key=feature)

    # styles features
    selected_styles = st.selectbox("Styles", options=styles_features, index=0)
    for feature in styles_features:
        inputs[feature] = 1 if feature == selected_styles else 0

    # Brand features
    selected_brand = st.selectbox("Brand", options=brand_features)
    for feature in brand_features:
        inputs[feature] = 1 if feature == selected_brand else 0

    # Country features
    selected_country = st.selectbox("Country", options=country_features)
    for feature in country_features:
        inputs[feature] = 1 if feature == selected_country else 0

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
        
        result = classify(prediction[0])
        st.success(result)

if __name__ == '__main__':
    main()
