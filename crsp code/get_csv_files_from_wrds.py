import wrds

db = wrds.Connection(username='marcchen')

# treasury_data = db.get_table(library='crsp', table='mcti')
# stock_index_data = db.get_table(library='crsp', table='msi')

# treasury_data.to_csv('treasury_2021.csv')
# stock_index_data.to_csv('stock_indexes_2021.csv')

int_index_data = db.get_table(library='crsp', table='msci')


int_index_data.to_csv('int_index_data_2022.csv')

db.close()
