# Import packages
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# Open and read the alerts.log file
with open('alerts.log', 'r') as file:
    data = file.readlines()
    file.close()

# Extracting the relevant parts and creating a list of lists
formatted_data = []
for row in data:
    try:
        split1 = re.split(r'\[\*\*\] \[[0-9]\:[0-9]+\:[0-9]+\]', row)
        split2 = re.split(r'\[Priority: [0-9]\]', split1[1])
        split3 = split2[1].split()

        date_time = split1[0].strip()
        message = split2[0].strip()
        protocol = split3[0].strip()
        source = split3[1].strip().split(':')[0]
        source_port = split3[1].strip().split(':')[1]
        dest = split3[3].strip().split(':')[0]
        dest_port = split3[3].strip().split(':')[1]

        formatted_data.append([date_time, message, protocol.strip('{}'), source, source_port, dest, dest_port])
    except Exception as e:
        # If something goes wrong just print the row where it went wrong and the error then quit
        # Will need to fix the error and rerun the script
        print("Error in row: \n", row)
        print("\nError: ", e)
        break

# Creating DataFrame
df = pd.DataFrame(formatted_data, columns=['Date_Time', 'Message', 'Protocol', 'Source', 'Source Port', 'Destination', 'Destination Port'])

# Create the label encoders for each data
le_protocol = LabelEncoder()
le_message = LabelEncoder()
le_IPs = LabelEncoder()
le_ports = LabelEncoder()

# Concat the IP's and ports to fit the label encoders to both collumns
temp_IP_list = pd.concat([df['Destination'], df['Source']])
temp_port_list = pd.concat([df['Source Port'], df['Destination Port']])

# Fit the label encoders to the IPs and Ports
le_IPs.fit(temp_IP_list)
le_ports.fit(temp_port_list)

# fit transform (and transform) all the collumns
df['Protocol'] = le_protocol.fit_transform(df['Protocol'])
df['Message'] = le_message.fit_transform(df['Message'])
df[['Source', 'Destination']] = df[['Destination', 'Source']].apply(lambda x: le_IPs.transform(x))
df[['Source Port', 'Destination Port']] = df[['Destination Port', 'Source Port']].apply(lambda x: le_ports.transform(x))

# Print the mapping counts for each label encoder
print("Mapping count protocol: ", len(le_protocol.classes_))
print("Mapping count message: ", len(le_message.classes_))
print("Mapping count IP's: ", len(le_IPs.classes_))
print("Mapping count port: ", len(le_ports.classes_))

# To write the labels to a file we need the list to be of strings
stringified_results = []

for index, row in df.iterrows():
    stringified_results.append(f"{row['Message']} ")

# Write the stringified results to a file that can be used as input for DeepCASE
with open('selfDataSet.txt', 'w') as file:
    file.writelines(stringified_results)
    file.close()