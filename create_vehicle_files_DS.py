import csv
import os

# create csv file for capacity variation of vehicles

vehicle_cap_max = 20
for n in range(1, vehicle_cap_max + 1):


    path_name = os.path.join("data", "vehicles", f'vehtype_cap_var_{n}.csv')
    os.remove(path_name)

    with open(path_name, 'a', newline='') as csvfile:
        fieldnames = ['vtype_name_full', f'vehtype_cap_var_{n}']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'vtype_name_full': 'maximum_passengers', f'vehtype_cap_var_{n}': f'{n}'})
        writer.writerow({'vtype_name_full': 'daily_fix_cost [cent]', f'vehtype_cap_var_{n}': '2500'})
        writer.writerow({'vtype_name_full': 'per_km_cost [cent]', f'vehtype_cap_var_{n}': '25'})
        writer.writerow({'vtype_name_full': 'battery_size [kWh]', f'vehtype_cap_var_{n}': '50'})
        writer.writerow({'vtype_name_full': 'range [km]', f'vehtype_cap_var_{n}': '10000'})
        writer.writerow({'vtype_name_full': 'source', f'vehtype_cap_var_{n}': ''})


# create csv file for range variation of vehicles

vehicle_range_max = 20
for n in range(1, vehicle_range_max + 1):


    path_name = os.path.join("data", "vehicles", f'vehtype_range_var_{n}.csv')
    #os.remove(path_name)

    with open(path_name, 'a', newline='') as csvfile:
        fieldnames = ['vtype_name_full', f'vehtype_range_var_{n}']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'vtype_name_full': 'maximum_passengers', f'vehtype_range_var_{n}': '4'})
        writer.writerow({'vtype_name_full': 'daily_fix_cost [cent]', f'vehtype_range_var_{n}': '2500'})
        writer.writerow({'vtype_name_full': 'per_km_cost [cent]', f'vehtype_range_var_{n}': '25'})
        writer.writerow({'vtype_name_full': 'battery_size [kWh]', f'vehtype_range_var_{n}': '50'})
        writer.writerow({'vtype_name_full': 'range [km]', f'vehtype_range_var_{n}': f'{n*10}'})
        writer.writerow({'vtype_name_full': 'source', f'vehtype_range_var_{n}': ''})