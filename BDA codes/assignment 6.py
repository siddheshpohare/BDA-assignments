import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Sample transactional data (customer, product, transaction_id)
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2'],
    'product_id': ['P1', 'P2', 'P3', 'P2', 'P3'],
    'amount': [100, 200, 150, 300, 120],
    'location': ['NY', 'CA', 'TX', 'NY', 'CA']
}

df = pd.DataFrame(data)

# Create a bipartite graph: customers (set 0), products (set 1)
G = nx.Graph()
G.add_nodes_from(df['customer_id'], bipartite=0)
G.add_nodes_from(df['product_id'], bipartite=1)
G.add_edges_from([(row['customer_id'], row['product_id']) for _, row in df.iterrows()])

# Visualize the bipartite graph
pos = nx.spring_layout(G, seed=42)  # Seed for reproducible layout
plt.figure(figsize=(8, 6))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color='skyblue',
    edge_color='gray',
    node_size=2000,
    font_size=12
)
plt.title("Customer-Product Transaction Graph")
plt.show()


import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

# Sample transactional data
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2'],
    'product_id': ['P1', 'P2', 'P3', 'P2', 'P3'],
    'amount': [100, 200, 150, 300, 120],
    'location': ['NY', 'CA', 'TX', 'NY', 'CA']
}
df = pd.DataFrame(data)

# Start Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.H2("Transactional Data Dashboard"),

    html.Label("Filter by Location:"),
    dcc.Dropdown(
        id='location-filter',
        options=[{'label': loc, 'value': loc} for loc in df['location'].unique()],
        value='NY'
    ),

    html.Br(),
    html.Div(id='summary'),

    dcc.Graph(id='bar-chart'),

    html.H4("Customer vs Product Crosstab"),
    html.Div(id='crosstab-table')
])

# Callback to update dashboard
@app.callback(
    [
        Output('bar-chart', 'figure'),
        Output('summary', 'children'),
        Output('crosstab-table', 'children')
    ],
    [Input('location-filter', 'value')]
)
def update_dashboard(selected_location):
    filtered = df[df['location'] == selected_location]

    # Summary
    total_sales = filtered['amount'].sum()
    summary = f"Total Sales in {selected_location}: ${total_sales}"

    # Bar Chart
    fig = px.bar(
        filtered,
        x='customer_id',
        y='amount',
        color='product_id',
        barmode='group',
        title="Customer Purchases by Product"
    )

    # Crosstab
    cross = pd.crosstab(
        filtered['customer_id'],
        filtered['product_id'],
        values=filtered['amount'],
        aggfunc='sum',
        margins=True
    ).fillna(0)

    # Build HTML Table
    table_header = [html.Th("Customer")] + [html.Th(col) for col in cross.columns]
    table_rows = [
        html.Tr([html.Td(row)] + [html.Td(cross.loc[row][col]) for col in cross.columns])
        for row in cross.index
    ]

    table = html.Table([html.Thead(html.Tr(table_header)), html.Tbody(table_rows)])

    return fig, summary, table

# Run server
if __name__ == '__main__':
    app.run(debug=True)

import pandas as pd
import matplotlib.pyplot as plt

# Sample transactional data
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2'],
    'product_id': ['P1', 'P2', 'P3', 'P2', 'P3'],
    'amount': [100, 200, 150, 300, 120],
    'location': ['NY', 'CA', 'TX', 'NY', 'CA']
}
df = pd.DataFrame(data)

# Export to CSV
df.to_csv("transaction_report.csv", index=False)

# Export to Excel
df.to_excel("transaction_report.xlsx", index=False)

# Export to XML
df.to_xml("transaction_report.xml", index=False)

# Export to PDF (using matplotlib)
fig, ax = plt.subplots(figsize=(8, 5))
df.groupby('customer_id')['amount'].sum().plot(kind='bar', ax=ax, color='skyblue')
plt.title('Total Amount by Customer')
plt.xlabel('Customer ID')
plt.ylabel('Amount ($)')
plt.tight_layout()
fig.savefig("transaction_report.pdf")
plt.close()

print("All reports exported successfully!")
