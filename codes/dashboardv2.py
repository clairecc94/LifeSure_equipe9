import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import plotly.figure_factory as ff

# Charger les données
df_merged = pd.read_csv("df_merged.csv")
df_merged1 = pd.read_csv("df_merged1.csv")

# Définition de la fonction Cramér's V pour les variables catégoriques
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    
    # Corriger r_corr et k_corr
    r_corr = r - ((r-1)**2) / (n-1)
    k_corr = k - ((k-1)**2) / (n-1)
    
    # Assurer que r_corr-1 et k_corr-1 ne soient pas égaux à zéro
    if (r_corr-1) <= 0 or (k_corr-1) <= 0:
        return np.nan  # Retourner NaN ou une valeur alternative pour éviter la division par zéro
    
    return np.sqrt(phi2corr / min(r_corr-1, k_corr-1))

num_cols = df_merged.select_dtypes(include=['number']).columns
cat_cols = df_merged.select_dtypes(include=['category', 'object']).columns
corr_matrix = pd.DataFrame(index=num_cols.union(cat_cols), columns=num_cols.union(cat_cols))

for col1 in corr_matrix.columns:
    for col2 in corr_matrix.index:
        if col1 == col2:
            corr_matrix.loc[col1, col2] = 1
        elif col1 in num_cols and col2 in num_cols:
            corr_matrix.loc[col1, col2] = df_merged[col1].corr(df_merged[col2])  # Pearson
        elif col1 in cat_cols and col2 in cat_cols:
            corr_matrix.loc[col1, col2] = cramers_v(df_merged[col1], df_merged[col2])  # Cramér's V
        else:
            corr_matrix.loc[col1, col2] = cramers_v(df_merged[col1].astype(str), df_merged[col2].astype(str))  # mix

corr_matrix = corr_matrix.astype(float)

# Calcul de la matrice de corrélation entre les variables spécifiques
corr_interactions = df_merged1[['charges', 'bmi_aqi', 'smoker_aqi', 'bmi_smoker_aqi']].corr(method='spearman')

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Mise en page du dashboard
app.layout = html.Div([
    html.H1("📊 LifeSure - Dashboard "),
    
    html.Label("Choisissez une variable"),
    dcc.Dropdown(
        id='variable-selector',
        options=[{'label': col, 'value': col} for col in df_merged.columns],
        value='charges'
    ),
    
    dcc.Graph(id='variable-graph'),

    html.Hr(),
    html.H3("Coût Médical par Groupe d'Âge"),
    dcc.Graph(id='boxplot-age-charges'),

    html.Hr(),
    html.H3("Coût Médical par Nombre d'Enfants"),
    dcc.Graph(id='boxplot-children-charges'),
    
    html.Hr(),
    html.H3("Charges Médicales par Fumeur / Non-Fumeur"),
    dcc.Graph(id='boxplot-smoker-charges'),

    html.Hr(),
    html.H3("Taux de Fumeurs dans Chaque Groupe d'Âge"),
    dcc.Graph(id='barplot-smoker-age'),

    html.Hr(),
    html.H3("Coût Médical par État"),
    dcc.Graph(id='barplot-state-charges'),
    
    html.Hr(),
    html.H3("Distribution AQI par État"),
    dcc.Graph(id='boxplot-state-aqi'),
    
    html.Hr(),
    html.H3("Heatmap des Corrélations Mixtes"),
    dcc.Graph(id='heatmap'),
    
    html.Hr(),
    html.H3("Heatmap des Corrélations entre Charges et Interactions AQI - Facteurs de Santé"),
    dcc.Graph(id='heatmap-aqi-interactions'),
    
    html.Hr(),
    html.H3("Distribution des Coûts Médicaux selon l'IMC et le Tabagisme"),
    dcc.Graph(id='bmi-smoker-boxplot'),

    html.Hr(),
    html.H3("Distribution des Coûts Médicaux selon la Pollution (AQI)"),
    dcc.Graph(id='boxplot-aqi-charges')
])

# Callback pour charges médicales par groupe d'age
@app.callback(
    Output('boxplot-age-charges', 'figure'),
    Input('variable-selector', 'value')
)
def update_boxplot_age_charges(variable):
    fig = px.box(df_merged, x='age_group', y='charges',
                 title="Coût Médical par Groupe d'Âge",
                 labels={'age_group': 'Groupe d\'Âge', 'charges': 'Coût Médical'},
                 color='age_group')
    return fig

# Callback pour charges médicales par nombre d'enfants
@app.callback(
    Output('boxplot-children-charges', 'figure'),
    Input('variable-selector', 'value')
)
def update_boxplot_children_charges(variable):
    fig = px.box(df_merged, x='children', y='charges',
                 title="Coût Médical par Nombre d'Enfants",
                 labels={'children': 'Nombre d\'Enfants', 'charges': 'Coût Médical'},
                 color='children')
    return fig

# Callback pour afficher le boxplot des charges médicales par fumeur/non-fumeur
@app.callback(
    Output('boxplot-smoker-charges', 'figure'),
    Input('variable-selector', 'value')
)
def update_smoker_charges_boxplot(variable):
    fig = px.box(df_merged, x='smoker', y='charges',
                 title="Charges Médicales par Fumeur / Non-Fumeur",
                 labels={'smoker': 'Fumeur / Non-Fumeur', 'charges': 'Charges Médicales'})
    return fig

# Callback taux de fumeurs par groupe d'âge
@app.callback(
    Output('barplot-smoker-age', 'figure'),
    Input('variable-selector', 'value')
)
def update_barplot_smoker_age(variable):
    df_smoker_rate = df_merged.groupby('age_group')['smoker'].mean() * 100 
    
    fig = px.bar(df_smoker_rate.reset_index(), 
                 x='age_group', 
                 y='smoker',
                 title="Taux de Fumeurs dans Chaque Groupe d'Âge",
                 labels={'smoker': 'Taux de Fumeurs (%)', 'age_group': 'Groupe d\'Âge'},
                 color_discrete_sequence=['#636EFA'])
    return fig

#Callback Cout medical par Etat
@app.callback(
    Output('barplot-state-charges', 'figure'),
    Input('variable-selector', 'value')
)
def update_boxplot_state_charges(variable):
    df_avg_charges = df_merged.groupby('state_name')['charges'].mean().reset_index()
    df_avg_charges.rename(columns={'charges': 'avg_charges'}, inplace=True)
    fig = px.bar(df_avg_charges, x='state_name', y='avg_charges',
                 title="Moyenne du Coût Médical par État",
                 labels={'state_name': 'État', 'avg_charges': 'Moyenne du Coût Médical'},
                 color_discrete_sequence=['#636EFA'])
    fig.update_xaxes(tickangle=60)
    return fig

# Callback pour afficher le boxplot AQI par état
@app.callback(
    Output('boxplot-state-aqi', 'figure'),
    Input('variable-selector', 'value')
)
def update_aqi_state_boxplot(variable):
    state_order = sorted(df_merged['state_name'].unique())
    fig = px.box(df_merged, x='state_name', y='aqi',
                 title="Distribution AQI par État",
                 labels={'state_name': 'État', 'aqi': 'Indice de qualité de l\'air (AQI)'},
                 category_orders={'state_name': state_order})
    fig.update_layout(xaxis_title="État", yaxis_title="AQI")
    fig.update_xaxes(tickangle=60)  
    return fig

# Callback pour afficher les graphiques dynamiques
@app.callback(
    Output('variable-graph', 'figure'),
    Input('variable-selector', 'value')
)
def update_graph(variable):
    # Vérifier si 'charges' est sélectionné
    if variable == 'charges':
        # Si 'charges' est sélectionné, afficher un histogramme (ou un boxplot)
        fig = px.histogram(df_merged, x='charges', nbins=50, title="Distribution des Charges Médicales")
        fig.update_layout(
            xaxis_title='Charges',
            yaxis_title='Fréquence'
        )
    else:
        # Sinon, afficher un bar chart pour les autres variables
        df_counts = df_merged[variable].value_counts().reset_index()
        df_counts.columns = ['category', 'count']
        
        # Créer le graphique
        fig = px.bar(df_counts, x='category', y='count')
        
        fig.update_layout(
            title=f"Distribution de {variable}",
            xaxis_title=variable,
            yaxis_title='Count'
        )
    
    # Afficher le graphique
    return fig

# Callback pour afficher la heatmap des corrélations mixtes
@app.callback(
    Output('heatmap', 'figure'),
    Input('variable-selector', 'value')
)
def generate_heatmap(variable):
    # Création de la heatmap avec Plotly
    rounded_corr_matrix = corr_matrix.round(4)
    fig = ff.create_annotated_heatmap(
        z=rounded_corr_matrix.values,
        x=list(rounded_corr_matrix.columns),
        y=list(rounded_corr_matrix.index),
        colorscale='RdBu_r',
        showscale=True
    )
    fig.update_layout(title="Heatmap des Corrélations Mixtes (Numériques et Catégoriques)")
    return fig

# Callback pour afficher la heatmap des corrélations AQI et Charges
@app.callback(
    Output('heatmap-aqi-interactions', 'figure'),
    Input('variable-selector', 'value')
)
def generate_aqi_interactions_heatmap(variable):
    # Créer la heatmap pour les interactions AQI et Charges avec Plotly
    fig = ff.create_annotated_heatmap(
        z=corr_interactions.values,
        x=list(corr_interactions.columns),
        y=list(corr_interactions.index),
        colorscale='RdBu',
        showscale=True
    )
    fig.update_layout(title="Corrélation entre Charges et Interactions AQI - Facteurs de Santé")
    return fig

# Callback pour afficher le boxplot IMC / Tabagisme / Charges
@app.callback(
    Output('bmi-smoker-boxplot', 'figure'),
    Input('variable-selector', 'value')
)
def update_boxplot(variable):
    fig = px.box(df_merged, x='smoker', y='charges', color='bmi_category',
                 title="Distribution des Coûts Médicaux selon l'IMC et le Tabagisme")
    return fig

# Callback pour afficher un boxplot AQI vs Charges
@app.callback(
    Output('boxplot-aqi-charges', 'figure'),
    Input('variable-selector', 'value')
)
def update_boxplot_aqi(variable):
    fig = px.box(df_merged, x='aqi', y='charges', color='aqi',
                 title="Distribution des Coûts Médicaux selon la Pollution (AQI)")
    return fig

# Lancer l'application
if __name__ == '__main__':
    app.run_server(debug=True)
