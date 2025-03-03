from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np
from shiny import App, render_image, ui, render, reactive
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.datasets import fetch_california_housing
from prettytable import PrettyTable
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error

# Estilos CSS personalizados
# Esta parte es interesante porque muestra cómo se puede mejorar la estética de la 
# aplicación web. En un entorno profesional, esto puede hacer que nuestras 
# aplicaciones sean más atractivas y fáciles de usar para los clientes.
custom_css = """
<style>
    .app-title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
    }
    .pretty-table {
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;s
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    .pretty-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: left;
    }
    .pretty-table th,
    .pretty-table td {
        padding: 12px 15px;
    }
    .pretty-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .pretty-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .pretty-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
</style>
"""

# Funcion para darle estilo a la web
# Esta función es muy útil para presentar los datos de manera legible. 
# Es un buen ejemplo de cómo podemos adaptar las salidas de nuestros análisis 
# para que sean más comprensibles para usuarios no técnicos.
def create_pretty_table(df, max_rows=10):
    table = PrettyTable()
    
    # Convertir el DataFrame a string para evitar problemas de tipo
    df_str = df.astype(str)
    
    # Manejar índices multinivel
    if isinstance(df.index, pd.MultiIndex):
        index_names = df.index.names
    else:
        index_names = [df.index.name if df.index.name else '']
    
    # Configurar los nombres de las columnas
    table.field_names = index_names + list(df.columns)
    
    # Añadir filas
    for i, (idx, row) in enumerate(df_str.iterrows()):
        if i >= max_rows:
            break
        # Manejar índices multinivel
        if isinstance(idx, tuple):
            table.add_row(list(idx) + list(row))
        else:
            table.add_row([idx] + list(row))
    
    return table.get_html_string(attributes={"class": "pretty-table"})

# Cargar el conjunto de datos
# Función cargar_dataset
# Aquí se cargan varios conjuntos de datos conocidos en el campo de la ciencia de datos. 
def cargar_dataset():
    california = fetch_california_housing()
    california_df = pd.DataFrame(california.data, columns=california.feature_names)
    california_df['MedHouseValue'] = california.target

    boston_df = pd.read_csv(Path(__file__).parent / "boston.csv")

    datasets = {
        "Iris": sns.load_dataset('iris'),
        "Tips": sns.load_dataset('tips'), 
        "Penguins" : pd.read_csv(Path(__file__).parent / "penguins.csv", na_values="NA"),
        "California": california_df,
        "Boston": boston_df,
    }
    return datasets

datasets = cargar_dataset()

# Crear la interfaz de usuario (agregando el CSS personalizado)
# Interfaz de usuario (UI)
# La creación de la interfaz de usuario con Shiny es bastante intuitiva. 
# Me gusta cómo se organiza todo en pestañas, lo que permite a los usuarios 
# explorar diferentes aspectos del análisis de forma ordenada.
# Todo debidamente comentado
app_ui = ui.page_fluid( 
    ui.head_content(ui.HTML(custom_css)),
    ui.div(
        ui.HTML('<h3 class="app-title">Análisis de Random Forest</h3>'),
        style="margin-bottom: 15px;"
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "datos", 
                "Selecciona los datos", 
                choices=list(datasets.keys())
            ),
            ui.input_radio_buttons(  
                "radio",  
                "Modelo",  
                {"1": "Clasificación", "2": "Regresión"},  
            ),  
            ui.input_select(
                "var_x", "Variable de Respuesta", choices=[]
            ),
            ui.input_select(  
                "explanatory",  
                "Variables Explicativas",  
                choices=[],  
                multiple=True,  
            ), 
            ui.input_numeric("mtry", "Mtry", 1, min=1, max=100), 
            ui.input_numeric("trees", "Árboles", 100, min=1, max=1000), 
            ui.input_numeric("seed", "Semilla", 42, min=1),   
            ui.input_action_button("train", "Entrenar Modelo", class_="btn-primary"),
        ),
        ui.h4("Entrenamiento y Análisis con Random Forest"),
        ui.navset_tab(
            ui.nav_panel(
                "Resumen",
                ui.output_ui("data_summary"),
            ),
            ui.nav_panel(
                "Información",
                ui.output_ui("data_info"),
            ),
            ui.nav_panel(
                "Entrenamiento",
                ui.output_ui("entrenamiento")
            ),
            ui.nav_panel(
                "Plots",
                ui.input_select(
                    "plot_type", "Tipo de Gráfico", 
                    choices=["Importancia de las Variables", "Dependencia Parcial"]
                ),
                ui.input_action_button("plot_generate", "Generar Gráfico", class_="btn-secondary"),
                ui.output_plot("plot_output"),
            ),
            ui.nav_panel(
                "Curva de Aprendizaje",
                ui.input_action_button("generate_learning_curve", "Generar Curva de Aprendizaje", class_="btn-secondary"),
                ui.output_plot("learning_curve_plot"),
            ),
            ui.nav_panel(
                "Visualización de Árbol",
                ui.input_numeric("tree_depth", "Profundidad máxima del árbol", 3, min=1, max=10),
                ui.input_action_button("visualize_tree", "Visualizar Árbol", class_="btn-secondary"),
                ui.output_plot("tree_visualization"),
            ),
            ui.nav_panel(
                "Optimización de Hiperparámetros",
                ui.input_numeric("n_iter", "Número de iteraciones", 10, min=1, max=100),
                ui.input_action_button("optimize_hyperparams", "Optimizar Hiperparámetros", class_="btn-secondary"),
                ui.output_text("optimization_results"),
            ),
            ui.nav_panel(
                "Análisis OOB",
                ui.input_numeric("max_trees_oob", "Número máximo de árboles para OOB", 100, min=10, max=500),
                ui.input_action_button("analyze_oob", "Analizar OOB Error", class_="btn-secondary"),
                ui.output_plot("oob_plot"),
            ),
        )
    ),
)

# Lógica del servidor
# Aqui estan las funciones
def server(input, output, session):
    # Aquí es donde ocurre la magia. Cada función decorada con @output
    # corresponde a un elemento reactivo en la UI
    # parte mas importante del desarrollo del proyecto.

    # Declaraciones de valores reactivos
    best_params = reactive.Value({})

    @reactive.Effect
    def update_variable_selectors():
        # Esta función actualiza dinámicamente las opciones de variables
        # basándose en el dataset seleccionado. Muy útil para la interactividad.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        columnas = list(dataset.columns)
        ui.update_select("var_x", choices=columnas)
        ui.update_select("explanatory", choices=columnas)
    
    @output
    @render.ui
    def data_summary():
        # Proporciona un resumen rápido de los datos. Esencial para el EDA.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        return ui.HTML(create_pretty_table(dataset))
    
    @output
    @render.ui
    def data_info():
        # Ofrece información más detallada sobre el dataset. 
        # Crucial para entender la estructura de los datos antes de modelar.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        
        info = ui.HTML(f"""
        <h4>Información del Dataset: {dataset_name}</h4>
        <p>Número de filas: {dataset.shape[0]}</p>
        <p>Número de columnas: {dataset.shape[1]}</p>
        <h5>Resumen Estadístico:</h5>
        {create_pretty_table(dataset.describe())}
        <h5>Tipos de Datos:</h5>
        {create_pretty_table(dataset.dtypes.to_frame(name='Tipo de Dato'))}
        <h5>Valores Faltantes:</h5>
        {create_pretty_table(dataset.isnull().sum().to_frame(name='Valores Faltantes'))}
        """)
        return info

    
    @output
    @render.ui
    @reactive.event(input.train)
    def entrenamiento():
        # Esta función es el corazón del análisis. Entrena el modelo Random Forest
        # y proporciona métricas de evaluación. Es interesante ver cómo se manejan
        # tanto la clasificación como la regresión en la misma función.
        dataset_name = input.datos()
        var_x = input.var_x()
        explanatory_vars = list(input.explanatory())
        trees = input.trees()
        seed = input.seed()
        radio = input.radio()
        mtry = input.mtry()

        dataset = datasets[dataset_name]
        X = dataset[explanatory_vars]
        y = dataset[var_x]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        if radio == "1":
            model = RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=mtry)
        else:
            model = RandomForestRegressor(n_estimators=trees, random_state=seed, max_features=mtry)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Validación cruzada
        cv_scores = cross_val_score(model, X, y, cv=5)

        if radio == "1":
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            evaluation_summary = (
                f"<h5>Reporte de Clasificación:</h5>"
                f"{create_pretty_table(report_df)}"
                f"<h5>Validación Cruzada (5-fold):</h5>"
                f"<p>Precisión media: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})</p>"
            )
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            evaluation_summary = (
                f"<h5>Métricas de Regresión:</h5>"
                f"<p>Error Cuadrático Medio (MSE): {mse:.4f}<br>"
                f"R² Score: {r2:.4f}</p>"
                f"<h5>Validación Cruzada (5-fold):</h5>"
                f"<p>R² medio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})</p>"
            )

        feature_importance = pd.DataFrame({
            'feature': explanatory_vars,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        summary = ui.HTML(f"""
        <h4>Resumen del Modelo</h4>
        <p><strong>Dataset:</strong> {dataset_name}<br>
        <strong>Tipo de Modelo:</strong> {"Clasificación" if radio == "1" else "Regresión"}<br>
        <strong>Variable de respuesta:</strong> {var_x}<br>
        <strong>Variables explicativas:</strong> {', '.join(explanatory_vars)}<br>
        <strong>Mtry:</strong> {mtry}<br>
        <strong>Número de árboles:</strong> {trees}<br>
        <strong>Semilla:</strong> {seed}</p>
        {evaluation_summary}
        <h5>Importancia de las Variables:</h5>
        {create_pretty_table(feature_importance)}
        """)
        
        return summary

    @output
    @render.plot
    @reactive.event(input.plot_generate)
    def plot_output():
        # Genera visualizaciones importantes como la importancia de las variables
        # y los gráficos de dependencia parcial. Estos son cruciales para 
        # interpretar el modelo Random Forest.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        explanatory_vars = list(input.explanatory())
        var_x = input.var_x()
        trees = input.trees()
        seed = input.seed()
        radio = input.radio()
        mtry = input.mtry()

        X = dataset[explanatory_vars]
        y = dataset[var_x]
        
        if radio == "1":
            model = RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=mtry)
        else:
            model = RandomForestRegressor(n_estimators=trees, random_state=seed, max_features=mtry)

        model.fit(X, y)

        if input.plot_type() == "Importancia de las Variables":
            importances = model.feature_importances_
            sorted_idx = importances.argsort()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([explanatory_vars[i] for i in sorted_idx])
            ax.set_xlabel("Importancia de las Variables")
            ax.set_title("Importancia de las Variables en el Modelo")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            PartialDependenceDisplay.from_estimator(model, X, features=explanatory_vars[:2], ax=ax)
            ax.set_title("Dependencia Parcial de las Variables")
        
        return fig

    # En la parte del servidor
    @output
    @render.plot
    @reactive.event(input.generate_learning_curve)
    def learning_curve_plot():
        # La curva de aprendizaje es fundamental para diagnosticar problemas
        # de sesgo y varianza. Es genial ver esto implementado interactivamente.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        explanatory_vars = list(input.explanatory())
        var_x = input.var_x()
        trees = input.trees()
        seed = input.seed()
        radio = input.radio()
        mtry = input.mtry()

        X = dataset[explanatory_vars]
        y = dataset[var_x]
        
        if radio == "1":
            model = RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=mtry)
        else:
            model = RandomForestRegressor(n_estimators=trees, random_state=seed, max_features=mtry)

        return plot_learning_curve(model, X, y)
    
    # Funciones auxiliares como plot_learning_curve, visualize_tree, etc.
    # Estas funciones muestran cómo podemos descomponer tareas complejas
    # en funciones más pequeñas y manejables.
    # Función para generar curvas de aprendizaje
    def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1):
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, 
            train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy" if isinstance(estimator, RandomForestClassifier) else "r2"
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel("Tamaño del conjunto de entrenamiento")
        ax.set_ylabel("Puntuación")
        ax.grid()
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Puntuación de entrenamiento")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Puntuación de validación cruzada")
        ax.legend(loc="best")
        ax.set_title("Curva de Aprendizaje")
        return fig

    # Función para visualizar un árbol del Random Forest
    def visualize_tree(model, feature_names, class_names=None, max_depth=3):
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(model.estimators_[0], 
                feature_names=feature_names, 
                class_names=class_names,
                filled=True, 
                rounded=True, 
                max_depth=max_depth,
                ax=ax)
        plt.title("Visualización de un árbol del Random Forest")
        return fig
    
    @output
    @render.plot
    @reactive.event(input.visualize_tree)
    def tree_visualization():
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        explanatory_vars = list(input.explanatory())
        var_x = input.var_x()
        trees = input.trees()
        seed = input.seed()
        radio = input.radio()
        mtry = input.mtry()
        max_depth = input.tree_depth()

        X = dataset[explanatory_vars]
        y = dataset[var_x]
        
        if radio == "1":
            model = RandomForestClassifier(n_estimators=trees, random_state=seed, max_features=mtry)
            class_names = list(dataset[var_x].unique().astype(str))
        else:
            model = RandomForestRegressor(n_estimators=trees, random_state=seed, max_features=mtry)
            class_names = None

        model.fit(X, y)
        return visualize_tree(model, explanatory_vars, class_names, max_depth)

    # En la parte del servidor, añade:
    @output
    @render.text
    @reactive.event(input.optimize_hyperparams)
    def optimization_results():
        # La optimización de hiperparámetros es crucial en el aprendizaje automático.
        # Aquí se implementa una búsqueda aleatoria, que es eficiente computacionalmente.
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        explanatory_vars = list(input.explanatory())
        var_x = input.var_x()
        n_iter = input.n_iter()
        radio = input.radio()

        X = dataset[explanatory_vars]
        y = dataset[var_x]
        
        param_distributions = {
            'n_estimators': sp_randint(10, 200),
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': sp_randint(10, 110),
            'min_samples_split': sp_randint(2, 21),
            'min_samples_leaf': sp_randint(1, 11),
            'bootstrap': [True, False]
        }
        
        if radio == "1":
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        optimized_params, best_score = optimize_hyperparameters(X, y, model, param_distributions, n_iter)
        
        # Almacenar los mejores parámetros
        best_params.set(optimized_params)
        
        # Acceder al contenido del valor reactivo para mostrarlo
        params_to_display = best_params.get()
        
        return f"Mejores parámetros encontrados:\n{params_to_display}\n\nMejor puntuación: {best_score:.4f}"
    
    def optimize_hyperparameters(X, y, estimator, param_distributions, n_iter=10, cv=5):
        random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=42,
        n_jobs=-1,
        scoring='r2' if isinstance(estimator, RandomForestRegressor) else 'accuracy'
    )
    
        random_search.fit(X, y)
        
        # Convertir los mejores parámetros a un diccionario regular
        best_params = dict(random_search.best_params_)
        
        return best_params, random_search.best_score_

    # Función para análisis de Out-of-Bag Error
    # Esta funcion ha cambiado al archivo que envie.
    def plot_oob_error(X, y, n_estimators, is_classifier, **kwargs):
        # El análisis OOB es una característica única de Random Forest.
        # Es genial ver cómo se implementa aquí para ayudar a determinar 
        # el número óptimo de árboles.
        estimators = range(1, n_estimators + 1, 1)
        errors = []

        if is_classifier:
            rf = RandomForestClassifier(n_estimators=n_estimators, oob_score=True, n_jobs=-1, **kwargs)
        else:
            rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, n_jobs=-1, **kwargs)
        
        rf.fit(X, y)

        for i in estimators:
            if is_classifier:
                # Para clasificación, usamos el OOB score directamente
                err = 1 - rf.oob_score_
            else:
                # Para regresión, calculamos el MSE usando predicciones de los primeros i árboles
                y_pred = np.mean([tree.predict(X) for tree in rf.estimators_[:i]], axis=0)
                err = mean_squared_error(y, y_pred)
            errors.append(err)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(estimators, errors)
        ax.set_xlabel("Número de árboles")
        ax.set_ylabel("Error Out-of-Bag")
        ax.set_title("Error Out-of-Bag vs Número de árboles")
        ax.set_ylim(bottom=0)
        return fig

    @output
    @render.plot
    @reactive.event(input.analyze_oob)
    def oob_plot():
        dataset_name = input.datos()
        dataset = datasets[dataset_name]
        explanatory_vars = list(input.explanatory())
        var_x = input.var_x()
        max_trees = input.max_trees_oob()
        radio = input.radio()

        X = dataset[explanatory_vars]
        y = dataset[var_x]
        
        return plot_oob_error(X, y, max_trees, radio == "1")

# Creación de la aplicación
app = App(app_ui, server)
# Esta última línea junta todo para crear la aplicación Shiny.
