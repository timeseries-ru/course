import numpy as np
import streamlit as st
import seaborn as sns

st.markdown("*Привет!* Я пример, созданный с помощью `streamlit`!")
samples = st.sidebar.slider("Количество", min_value=10, max_value=500, value=250, step=10)

np.random.seed(1)
sns.distplot(np.random.normal(size=samples))

st.pyplot()
