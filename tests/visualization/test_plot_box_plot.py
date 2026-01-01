import pandas as pd
import matplotlib.pyplot as plt
from visualization.explore_continuous import plot_box_plot

#--- Function : test_plot_box_plot_basic ---
def test_plot_box_plot_basic():
    df = pd.DataFrame({
        "age": [25, 30, 22, 40, 35],
        "salary": [50000, 60000, 45000, 80000, 75000]
    })
    
    plt.ioff()  # empêche l'affichage automatique
    plot_box_plot(df, ["age", "salary"])
    plt.ion()
    
    # Vérifications simples
    assert "age" in df.columns
    assert "salary" in df.columns

#--- Function : test_plot_box_plot_with_missing_column ---
def test_plot_box_plot_with_missing_column():
    df = pd.DataFrame({"age": [25, 30, 22]})
    
    plt.ioff()
    plot_box_plot(df, ["age", "height"])  # 'height' n'existe pas
    plt.ion()
    
    assert "age" in df.columns

#--- Function : test_plot_box_plot_empty_list ---
def test_plot_box_plot_empty_list():
    df = pd.DataFrame({"age": [25, 30, 22]})
    
    plt.ioff()
    plot_box_plot(df, [])  # liste vide
    plt.ion()
    
    assert df.shape[1] == 1
