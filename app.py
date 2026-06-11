import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Konfiguracja strony
st.set_page_config(page_title="System Analityczny Titanic", layout="wide")
sns.set_theme(style="whitegrid")

# Ładowanie danych z bazy SQL
@st.cache_data
def load_data():
    if not os.path.exists('titanic.db'):
        return None
    conn = sqlite3.connect('titanic.db')
    df = pd.read_sql_query('SELECT * FROM passengers', conn)
    conn.close()
    return df

df = load_data()

# Ładowanie modelu ML
@st.cache_resource
def load_model():
    if os.path.exists('titanic_model.pkl'):
        return joblib.load('titanic_model.pkl')
    return None

model = load_model()

# Główny Interfejs
st.title("🚢 Zaawansowany System Analityczny Titanic")
st.markdown("Lokalna aplikacja integrująca bazę danych SQLite, analizę EDA oraz predykcję ML.")

if df is None:
    st.error("Brak bazy danych 'titanic.db'. Uruchom najpierw skrypt 'etl_train.py' w terminalu!")
else:
    tab1, tab2, tab3 = st.tabs([" Dashboard i Statystyki", "🤖 Prognozowanie ML", "🗄️ Przeglądarka Bazy Danych"])

    # ZAKŁADKA 1: DASHBOARD
    with tab1:
        st.header("Ogólne statystyki pasażerów")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Liczba pasażerów w bazie", len(df))
        col2.metric("Ogólna przeżywalność", f"{df['Survived'].mean():.1%}")
        col3.metric("Średni wiek pasażera", f"{df['Age'].mean():.1f} lat")
        col4.metric("Średnia cena biletu", f"{df['Fare'].mean():.2f} $")
        
        st.write("---")
        st.subheader("Filtrowanie analizy szczegółowej")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pclass_filter = st.multiselect("Wybierz klasę podróży:", options=[1, 2, 3], default=[1, 2, 3])
        with col_f2:
            sex_filter = st.multiselect("Wybierz płeć:", options=["male", "female"], default=["male", "female"])
            
        filtered_df = df[df['Pclass'].isin(pclass_filter) & df['Sex'].isin(sex_filter)]
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            st.markdown("### Szanse na przeżycie według płci")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(x='Sex', y='Survived', data=filtered_df, ax=ax1, palette="pastel", errorbar=None)
            ax1.set_ylabel("Procent przeżycia")
            st.pyplot(fig1)
            
        with col_plot2:
            st.markdown("### Rozkład wieku pasażerów")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=filtered_df, x='Age', hue='Survived', multiple='dodge', kde=True, palette='coolwarm', ax=ax2)
            ax2.legend(title="Przeżycie", labels=["Przeżył", "Nie przeżył"])
            ax2.set_xlabel("Wiek")
            ax2.set_ylabel("Liczba osób")
            st.pyplot(fig2)

    # ZAKŁADKA 2: PREDAKCJA ML
    with tab2:
        st.header("🤖 Kalkulator Szans Przeżycia (Random Forest)")
        if model is None:
            st.warning("⚠️ Nie znaleziono pliku modelu 'titanic_model.pkl'. Uruchom najpierw skrypt 'etl_train.py'.")
        else:
            st.write("Wprowadź dane pasażera, aby sprawdzić prognozę modelu:")
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                pclass = st.selectbox("Klasa podróży (Pclass):", [1, 2, 3], index=0)
                sex = st.radio("Płeć (Sex):", ["Mężczyzna", "Kobieta"])
                age = st.slider("Wiek (Age):", min_value=0, max_value=100, value=30, step=1)
                
            with col_in2:
                fare = st.number_input("Opłata za bilet (Fare) w $:", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
                sibsp = st.number_input("Liczba rodzeństwa/małżonków na pokładzie:", min_value=0, max_value=10, value=0)
                parch = st.number_input("Liczba rodziców/dzieci na pokładzie:", min_value=0, max_value=10, value=0)
                
            family_size = sibsp + parch + 1
            sex_encoded = 1 if sex == "Kobieta" else 0
            
            input_data = pd.DataFrame([{
                'Pclass': pclass,
                'Sex_encoded': sex_encoded,
                'Age': age,
                'Fare': fare,
                'FamilySize': family_size
            }])
            
            st.write("---")
            if st.button("📊 Uruchom Analizę Predykcyjną", type="primary"):
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
                
                if prediction == 1:
                    st.success(f"### 🎉 PRZEWIDZIANO: Pasażer przeżyje!")
                    st.metric("Prawdopodobieństwo przeżycia", f"{probabilities[1]:.1%}")
                else:
                    st.error(f"### 💀 PRZEWIDZIANO: Pasażer prawdopodobnie NIE przeżyje.")
                    st.metric("Prawdopodobieństwo odejścia", f"{probabilities[0]:.1%}")

    # ZAKŁADKA 3: PRZEGLĄDARKA SQL
    with tab3:
        st.header("🗄_\ufe0f Podgląd tabeli `passengers` z SQLite")
        search_id = st.text_input("Filtruj po ID Pasażera (PassengerId):")
        if search_id:
            try:
                conn = sqlite3.connect('titanic.db')
                search_df = pd.read_sql_query(f'SELECT * FROM passengers WHERE PassengerId = {int(search_id)}', conn)
                conn.close()
                st.dataframe(search_df)
            except ValueError:
                st.error("Podaj poprawną liczbę jako ID!")
        else:
            st.dataframe(df.head(50), use_container_width=True)