1. **Pipeline ETL & Modelowanie ('etl_train.py'):**
  - **Czyszczenie Danych:** Imputacja brakujących wartości wieku (mediana) oraz portu zaokrętowania.
  - -**Feature Engineering:** Kodowanie zmiennych kategorycznych ('Sex_Encoded') oraz wyliczanie wielkości rodziny ('FamilySize = SibSp + Parch + 1').
  - --**Baza danych:** Zapis przetworzonego zbioru danych do realcyjnej bazy SQLite('titanic.db') do tabeli 'passengers'.
  - -**Trening ML:** Strojenie hiperparametrów modelu 'RandomForestClassifier' metodą przeszukiwania siatki ('GridSearchCV' z 5-krotną walidacją krzyżową). Najlepszy model eksportowany do pliku 'titanic_model.pkl'.

2. **Aplikacja Dashboard('app.py'):**
   - Lekki serwer webowy **Streamlit** wczytujący bazę danych SQLite oraz plik zoptymalizowanego modelu, oferujący interaktywne interfejsy dla użytkownika końcowego.
  
---

## Jak uruchomić projekt lokalnie: 

## 1. Przygotowanie środowiska
Niezbędnym jest, aby w folderze projektu znajdował się plik 'train.csv'

Instalacja niezbędnych pakietów
'''bash
pip install -r requrements.txt

