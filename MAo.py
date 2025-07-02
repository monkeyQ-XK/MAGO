import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from streamlit_option_menu import option_menu
from datetime import datetime
import os
from io import BytesIO

# ----------- Home Page -----------
def home_page():
    st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('img.png')

    st.subheader("Magnesium Alloys: Lightweight, Strong, and Innovative")

    st.markdown("""
    Magnesium alloys are among the lightest structural materials available, combining low density with high strength-to-weight ratios.  
    They are widely used in automotive, aerospace, electronics, and biomedical applications.  
    This tool uses machine learning models to predict key mechanical and electrochemical properties of magnesium alloys based on their composition and processing parameters.
    """)

    st.markdown("<hr style='border: 1px solid #ccc;'/>", unsafe_allow_html=True)

    st.markdown("""
    **Engineering Research Center for Nanomaterials (ERCN)**, Room 327  
    College of Chemistry and Molecular Sciences, Henan University  
    Kaifeng, Henan 475004, P.R. China
    """)

# ----------- Predict Page -----------
def predict_page():
    st.markdown("<h2 style='text-align: center; color: #3366cc;'>🔧 Magnesium Alloy Property Predictor</h2>", unsafe_allow_html=True)
    name = st.text_input("👤 Name", max_chars=30)



    # --------------------------
    # 元素输入部分
    # --------------------------
    elements = ['Mg', 'Al', 'Zn', 'Y', 'Zr', 'Ca', 'Mn', 'Li', 'Sn', 'Nd', 'Gd', 'La', 'Si',
                'Sm', 'Ce', 'Er', 'Cu', 'Sr', 'Bi', 'Sb', 'Pb', 'Ni', 'Fe']
    element_input = []

    st.markdown("#### 🧪 Element Composition (wt. fraction)")

    for row_start in range(0, len(elements), 6):
        row_elements = elements[row_start:row_start + 6]
        cols = st.columns(len(row_elements))
        for i, elem in enumerate(row_elements):
            with cols[i]:
                st.caption(f"**{elem}**")
                ratio = st.number_input(
                    label=f"{elem}",
                    min_value=0.0, max_value=1.0, value=0.0,
                    step=0.01,
                    key=f"elem_{elem}",
                    label_visibility="collapsed"
                )
                element_input.append(ratio)

    # --------------------------
    # 实验参数输入部分
    # --------------------------
    st.markdown("#### ⚙️ Experimental Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        M_co = st.number_input(
            "M_Co",
            min_value=0.0, max_value=3.0, value=0.0, step=1.0,
            key="mco"
        )
    with col2:
        T_ex = st.number_input(
            "T_ex",
            min_value=0.0, max_value=1000.0, value=0.0, step=1.0,
            key="tex"
        )
    with col3:
        Ra_ex = st.number_input(
            "Ra_ex",
            min_value=0.0, max_value=100.0, value=0.0, step=0.1,
            key="raex"
        )
    with col4:
        R_ex = st.number_input(
            "R_ex",
            min_value=0.0, max_value=100.0, value=0.0, step=0.1,
            key="rex"
        )
    st.markdown(
        "##### ℹ️ Notes: Ra_ex = Extrusion ratio (0 = as-cast), T_ex = Extrusion temperature (0 = as-cast), M_Co = Corrosive Medium (0:NaCl, 1:SBF,2:Hank’s,3:FBS), R_ex = Extension rate (mm/min)")
    if st.button("📈 Predict"):
        st.markdown("---")
        element_df = pd.DataFrame([element_input], columns=elements)
        st.write("#### 📋 Composition Input")
        st.dataframe(element_df)

        # Load models
        E_model = load('Exgb.joblib')
        UTS_model = load('UTSxgb.joblib')
        EL_model = load('ELxgb.joblib')

        def preprocess_data(X):
            df1 = pd.read_excel('proo.xlsx').iloc[:, 1:]
            X1 = np.dot(X, df1)
            X1 = pd.DataFrame(X1, columns=df1.columns.tolist())
            X2 = []
            for i in range(len(X)):
                f1 = np.square(df1 - pd.DataFrame(np.tile(X1.iloc[i,], (23, 1)), columns=df1.columns.tolist()))
                value = np.dot(X.iloc[i, :], f1.values)
                X2.append(value)
            X2 = pd.DataFrame(X2, columns=['∆' + col for col in df1.columns.tolist()])
            return pd.concat([X1, X2], axis=1)

        processed_data = preprocess_data(element_df)
        fixed = pd.DataFrame([[M_co, T_ex, Ra_ex, R_ex]], columns=['M_Co', 'T_ex ', 'Ra_ex', 'R_ex'])
        processed_data = pd.concat([processed_data, fixed], axis=1)

        # Standardization
        t1 = ['M_Co', '∆HF', '∆M2', '∆M3', 'HF', '∆VEC', 'I3', '∆Rc', 'BP']
        dfa = pd.read_csv('esellect.csv')
        sc1 = StandardScaler().fit(dfa[t1])
        f1_input = sc1.transform(processed_data[t1])

        t2 = ['T_ex ', 'Ra_ex', 'R_ex', 'H2', '∆HF', '∆H1', 'I2', 'SEN', '∆A2']
        dfb = pd.read_csv('UTSsellect.csv')
        sc2 = StandardScaler().fit(dfb[t2])
        f2_input = sc2.transform(processed_data[t2])

        t3 = ['Ra_ex', '∆I1', 'Rc', 'Cs', 'M1', 'I1', 'VEC', '∆P', 'SGN']
        dfc = pd.read_csv('ELsellect.csv')
        sc3 = StandardScaler().fit(dfc[t3])
        f3_input = sc3.transform(processed_data[t3])

        # Predict
        E_pred = E_model.predict(f1_input)
        UTS_pred = UTS_model.predict(f2_input)
        EL_pred = EL_model.predict(f3_input)

        st.markdown("#### ✅ Prediction Results")
        result_df = pd.DataFrame([[E_pred[0], UTS_pred[0], EL_pred[0]]],
                                 columns=['Ecorr (V)', 'UTS (MPa)', 'Elongation (%)'])
        st.dataframe(result_df.style.highlight_max(axis=1, color='lightgreen'))

        # Save history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_row = {
            "Name": name, "Time": timestamp,
            **{k: v for k, v in zip(elements, element_input)},
            "M_Co": M_co, "T_ex": T_ex, "Ra_ex": Ra_ex, "R_ex": R_ex,
            "Ecorr (V)": E_pred[0], "UTS (MPa)": UTS_pred[0], "Elongation (%)": EL_pred[0]
        }
        df_hist = pd.DataFrame([history_row])
        if os.path.exists("history.csv"):
            df_hist.to_csv("history.csv", mode="a", index=False, header=False)
        else:
            df_hist.to_csv("history.csv", index=False)

        # Visualization
        # Visualization
        st.markdown("#### 📊 3D Visualization")
        df_plot = pd.read_excel("数据 (2).xlsx")

        with st.expander("📊 Show 3D Visualization"):
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(df_plot['Ecorr'], df_plot['UTS'], df_plot['EL'], c='lightcoral', label='Dataset', alpha=0.6,
                       edgecolors='k')
            ax.scatter(E_pred, UTS_pred, EL_pred, c='deepskyblue', marker='^', s=120, label='Prediction',
                       edgecolors='black')

            ax.set_xlabel('Ecorr (V)', fontsize=10, labelpad=10)
            ax.set_ylabel('UTS (MPa)', fontsize=10, labelpad=10)
            ax.set_zlabel('Elongation (%)', fontsize=10, labelpad=10)
            ax.set_title("3D Property Prediction Plot", fontsize=12, pad=10)
            ax.view_init(elev=20, azim=-135)
            ax.legend()
            fig.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

            st.pyplot(fig)
            plt.close(fig)


# ----------- History Page -----------
def history_page():
    st.title("📚 Prediction History")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df, use_container_width=True)

        # Download as Excel
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='History')
        st.download_button("📤 Download as Excel", data=buffer.getvalue(),
                           file_name="prediction_history.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Clear history
        if st.button("🗑️ Clear All History"):
            os.remove("history.csv")
            st.success("History cleared.")
    else:
        st.info("No prediction history yet.")

# ----------- Contact Page -----------
def contact_page():
    st.title("📬 Contact Us")
    st.markdown("""
    **Email:** qixinke@henu.edu.cn  
    **Address:** Kaifeng, Henan 475004, P.R. China  
    **Lab:** Engineering Research Center for Nanomaterials, Room 327  
    Henan University
    """)

# ----------- App Launcher -----------
def main():
    st.set_page_config(page_title='Magnesium Alloy Predictor', layout='wide')
    selected = option_menu(None, ["Home", "Predict", "History", "Contact"],
                           icons=["house", "calculator", "clock-history", "envelope"],
                           menu_icon="cast", default_index=0, orientation="horizontal")

    if selected == "Home":
        home_page()
    elif selected == "Predict":
        predict_page()
    elif selected == "History":
        history_page()
    elif selected == "Contact":
        contact_page()

if __name__ == "__main__":
    main()
