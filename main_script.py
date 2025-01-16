
"""
streamlit run /tmp/test.py --server.port 80
"""

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import plotly.express as px
import seaborn as sns
import os
import traceback
import glob


import joblib

def sent(x):
    if x <=2:
        return 'негативные'
    elif x >3:
        return 'позитивные'
    return 'нейтральные'

@st.cache_data
def dowload_df():
    df = pd.read_excel('MyFin Revuews.xlsx')
    df['sent'] = df['count_stars'].apply(lambda x: sent(x))
    return df

def calculate_nps(x):
    return round(((x['stars_4']+x['stars_5']) - (x['stars_1']+x['stars_2']))/x['all_review']*100)

def revue_nsp(df):
    t = df.groupby(['bank_name','count_stars'])['id'].count().reset_index()
    df_stars = t.pivot_table(index='bank_name', columns='count_stars', values='id').reset_index()
    df_stars = df_stars.fillna(0)
    df_stars = df_stars.rename(columns = {1:'stars_1',2:'stars_2',3:'stars_3',4:'stars_4',5:'stars_5'})
    df_stars['all_review'] = df_stars.apply(lambda x: x['stars_4']+x['stars_5']+x['stars_1']+x['stars_2'],axis=1)
    df_stars['NPS'] = df_stars.apply(lambda x: calculate_nps(x),axis=1)
    return df_stars

st.set_page_config(
        page_title="ML NLP",
        # page_icon="chart_with_upwards_trend",
        page_icon="🦈",
        layout="wide",
    )

st.markdown("""<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous"> """,unsafe_allow_html=True)

# загрузка DataSet 
df = dowload_df()
bank_list = np.sort(df['bank_name'].unique())
# Загрузка модели и векторизатора
model_SVC = joblib.load('model_SVC.pkl')
tf_idf_vectorizer_n2 = joblib.load('tf_idf_vectorizer.pkl')

with open('style.css', mode='r', encoding='utf') as fl:
    style_css = fl.read()
st.markdown(style_css, unsafe_allow_html=True)
st.markdown(f'<p class="caption-font">ML NLP</p>', unsafe_allow_html=True)



with st.sidebar:
    selected = option_menu(
        "Меню", 
        ["Аналитика отзывов", 'Определение тональности',], 
        icons=['house', 'file-earmark-person','currency-dollar',''], 
        menu_icon="menu-up",
         styles={
            "nav-link": {"font-size": "13px"},
        }, 
        default_index=0
    )


if selected == 'Аналитика отзывов':
    st.markdown(f'<p class="alert alert-primary">{selected}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="text-uppercase display-6">Распределение количества отзывов по банкам</p>', unsafe_allow_html=True)
    st.divider()
    df_revue_destrib = df.groupby('bank_name').size().sort_values(ascending=False).reset_index(name='reviews')
    st.bar_chart(df_revue_destrib,x='bank_name',y='reviews',x_label ='Банки',y_label ='Количество отзывов')
    st.divider()
    st.markdown(f'<p class="text-uppercase display-6">Расчет показателя NPS на основе отзывов</p>', unsafe_allow_html=True)
    df_stars = revue_nsp(df)
    st.bar_chart(df_stars,x='bank_name',y='NPS',x_label ='Банки',y_label ='NPS',color='#ff9f9b')
    st.divider()
    st.markdown(f'<p class="text-uppercase display-6">Распределение отзывов по тональности</p>', unsafe_allow_html=True)
    bank_option = st.selectbox("",(bank_list),)
    df_count_stars = df[df['bank_name']==bank_option].groupby('sent').size().reset_index(name='cnt')
    df_count_stars['perc'] = df_count_stars['cnt'].apply(lambda x: round(x/df_count_stars['cnt'].sum()*100,2))
    fig = px.pie(df_count_stars, values='perc', names='sent',color='sent',color_discrete_map={'негативные':'#ef553b','нейтральные':'#636efa','позитивные':'#00cc96'})
    st.plotly_chart(fig, use_container_width=True)


if selected == 'Определение тональности':
    st.markdown(f'<p class="alert alert-primary">{selected}</p>', unsafe_allow_html=True)
    st.markdown(f'<p ">Для определения тональности введите текст и нажмите кнопку</p>', unsafe_allow_html=True)
    text_review = st.text_input("")
    if st.button("Определить тональность", type="secondary"):
        # Преобразование текста в вектор
        review_vector = tf_idf_vectorizer_n2.transform([text_review])
        # Предсказание тональности
        prediction = model_SVC.predict(review_vector)
        probs = model_SVC.predict_proba(review_vector)
        if prediction[0] == 1:
            st.markdown(f'<p class="text-success">Комментарий положительный</p>', unsafe_allow_html=True)
        elif prediction[0] == 0:
            st.markdown(f'<p class="text-danger">Комментарий отрицательный</p>', unsafe_allow_html=True)
        st.write(f"вероятность класса 0 (отрицательный) - {probs[:, 0][0]:.20f}")
        st.write(f"вероятность класса 1 (положительный) - {probs[:, 1][0]:.20f}")