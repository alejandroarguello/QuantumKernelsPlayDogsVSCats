from website import create_app

app = create_app()

if __name__ == '__main__': #esta linea indica que se ejecute lo de dentro del if solo si se ejecuta este main.py, no si se importa.
    app.run(debug=False) #empieza el server y debug True hace que se haga rerun del server con cada cambio en nuestro codigo