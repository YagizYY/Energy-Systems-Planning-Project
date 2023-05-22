# -*- coding: utf-8 -*-
"""
Created on Fri May 12 19:27:18 2023

@author: beyza-25 & yagiz-55
"""

#%%
import os   

#%%
import gurobipy as grb
import pandas as pd
import numpy as np
from gurobipy import quicksum
from gurobipy import GRB
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def format_func(x):
    return "{:.0f}".format(x)

# Set the formatter function globally for numpy
np.set_printoptions(formatter={'float_kind': format_func})

#%%
# PARAMETERS #

time = len(pd.read_excel("C:/Users/yagiz/Desktop/4-2/IE-453/Energy-Systems-Planning-Project/data/Veri_Solar_Demand.xlsx", sheet_name="Demand")["MWH"])

# length of time period
n = 3

# interest rate
i= 0.05

# dimensionless annualization parameter for hydropower station
# d_h = i/(1-(1+i)**(-60))
d_h = 0.053

# dimensionless annualization parameter for solarpower station
# d_s = i/(1-(1+i)**(-30))
d_s = 0.065

# dimensionless annualization parameter for transmission line
# d_t = i/(1-(1+i)**(-40))
d_t = 0.058

# percentage of power loss during transmission lines
l = 0.05

# acceleration
g = 9.8

# height of the reservoir in hydropower generation in meter
h = 100

# one way efficiency of hydropower stations both in generating and pumping mode
alpha = 0.88

# efficiency of solar panels
gamma = 0.12

# a large number
M = 99999999

# unit cost of reservoir capacity in hydropower generation $/m3
C_s = 3

# unit cost of generator capacity in hydropower generation $/kW
C_pg = 500

# unit cost of solar panel area $/m2
C_m = 200 

# unit cost of transmission line capacity  $/kWh -> $0.8*(310 km)/(kw*km*3h)=  $/KWh 82.5
C_t  = 82.67

# unit cost of generating electricity using diesel generator $/kWh
mu = 0.25

# density of water in kg/m3
d = 1000

# EXOGENOUS VARIABLES #

demand = pd.read_excel("C:/Users/yagiz/Desktop/4-2/IE-453/Energy-Systems-Planning-Project/data/Veri_Solar_Demand.xlsx", sheet_name="Demand")["MWH"]
demand = np.array(demand)
demand = demand*1000 # now it is in kWh

inflow = np.loadtxt("C:/Users/yagiz/Desktop/4-2/IE-453/Energy-Systems-Planning-Project/data/Chenab1970.txt")
inflow = np.concatenate((inflow[972:], inflow[0:972])) 
inflow = inflow*10**3


solar_rad = pd.read_excel("C:/Users/yagiz/Desktop/4-2/IE-453/Energy-Systems-Planning-Project/data/Veri_Solar_Demand.xlsx", sheet_name="Solar") 
solar_rad = solar_rad.iloc[:, 0].to_numpy() # kw/m2


solar_radiation = []

for i in range(0,len(solar_rad), 3):
    solar_radiation.append(sum(solar_rad[i:i+3]))

solar_radiation = np.concatenate((solar_radiation[972:], solar_radiation[0:972])) 


# CONVENTIONAL
# DECISION VARIABLES 

c_model = grb.Model()

# State Variables

# Water stored in the upper reservoir in hydropower generation point at the end of time t
SU_t = c_model.addVars(range(0, time), lb = 0, name="SU_t")

# water stored in the lower reservoir at point i at the end of time t
SL_t = c_model.addVars(range(0, time), lb = 0, name="SL_t")

# mismatched demand at the end of time t
Z_t = c_model.addVars(range(0, time), lb = 0, name="Z_t")

# Electricity sent from hydropower generation point to demand point at the end of time t
Tsd_t = c_model.addVars(range(0, time), lb = 0, name="Tsd_t")

# Electricity sent from demand point to hydropower generation at the end of time t
Tds_t = c_model.addVars(range(0, time), lb = 0, name="Tds_t")

# Water spilled from upper reservoir at point i at the end of time t
LU_t = c_model.addVars(range(0, time), lb = 0, name="LU_t")

# Water spilled from lower reservoir at point i at the end of time t
LL_t = c_model.addVars(range(0, time), lb = 0, name="LL_t")

# Solar energy internally used in point j at the end of time t
V_t = c_model.addVars(range(0, time), lb = 0, name="V_t")

# Water released from upper reservoir at point i at the end of time t
R_t = c_model.addVars(range(0, time), lb = 0, name="R_t")

# Water pumped from lower reservoir at point i at the end of time t
P_t = c_model.addVars(range(0, time), lb = 0, name="P_t")

# Upper reservoir capacity 
SUMax = c_model.addVar(lb=0, name="SUMax") 
# Removed vtype=grb.GRB.CONTINUOUS: By default, Gurobi assumes variables to be continuous, so you don't need to specify the variable type explicitly.

# Lower reservoir capacity 
SLMax = c_model.addVar(lb = 0, name="SLMax")

# Solar panel size 
SS = c_model.addVar(lb = 0, name="SS")

# Generator size 
PGMax = c_model.addVar(lb = 0, name="PGMax")

# Maximum energy (capacity) transmitted from generator to demand points
TMax = c_model.addVar(lb = 0, name="TMax")

# Ip - binary variable to prevent pump and release&spill happening at the same time. 
Ip = c_model.addVars(range(0, time), vtype=grb.GRB.BINARY, name="Ip")

# Isd - binary variable to prevent sending electricity to and from village
Isd = c_model.addVars(range(0, time), vtype=grb.GRB.BINARY, name="Isd")

# CONSTRAINTS

#(0)
c_model.addConstr((SLMax == 0))

#(1)
c_model.addConstrs((SL_t[t] <= SLMax)  for t in range(0, time))

#(2)
c_model.addConstrs((SU_t[t] <= SU_t[t-1]  + inflow[t] + P_t[t] - R_t[t] - LU_t[t]) for t in range(1, time))

#(3)
c_model.addConstr(SU_t[0] <=  SUMax + inflow[0] + P_t[0] - R_t[0] - LU_t[0])

#(4) # paperda bu yok!!!!!
c_model.addConstrs((SU_t[t] <= SUMax) for t in range(0, time))

#(5)
c_model.addConstrs((SL_t[t] == SL_t[t-1] + R_t[t] - P_t[t] - LL_t[t]) for t in range(1, time))

#(6)
c_model.addConstr(SL_t[0] <=  R_t[0] - P_t[0] - LL_t[0])

#(7) 
c_model.addConstrs((R_t[t]*d*g*h*alpha/3600000 <= PGMax*n) for t in range(0, time))

#(8)
c_model.addConstrs((P_t[t]*d*g*h/alpha/3600000 <= PGMax*n) for t in range(0, time))

#(9)
c_model.addConstrs((Tsd_t[t] == R_t[t]*d*g*h*alpha/3600000) for t in range(0, time))

#(10)
c_model.addConstrs((P_t[t]*d*g*h/alpha/3600000 == Tds_t[t]*(1-l)) for t in range(0, time))

#(11)
c_model.addConstrs((Tsd_t[t] <= TMax*n) for t in range(0, time))

#(12)
c_model.addConstrs((Tds_t[t] <= TMax*n) for t in range(0, time))

#(13)
c_model.addConstrs((solar_radiation[t]*SS*gamma*n >= V_t[t] + Tds_t[t])  for t in range(0, time))

#(14)
c_model.addConstrs((Z_t[t] == demand[t] - V_t[t] - Tsd_t[t]*(1-l)) for t in range(0, time))

#(15)
c_model.addConstrs((P_t[t] <= Ip[t]*M) for t in range(0, time))

#(16)
c_model.addConstrs((R_t[t] <= (1-Ip[t])*M) for t in range(0, time))

#(17)
c_model.addConstrs((LU_t[t] <= (1-Ip[t])*M) for t in range(0, time))

#(18)
c_model.addConstrs((Tsd_t[t] <= Isd[t]*M) for t in range(0, time))

#(19)
c_model.addConstrs((Tds_t[t] <= (1-Isd[t])*M) for t in range(0, time))

#(20)
c_model.addConstr(SU_t[time-1] == SUMax)

Objective = d_h*C_s*(SUMax+SLMax)  + d_h*C_pg*PGMax + d_s*SS*C_m + d_t*C_t*TMax + quicksum(Z_t[t]*mu*n for t in range(0, time))


c_model.setObjective(Objective, GRB.MINIMIZE)

c_model.optimize()


# PHES

p_model = grb.Model()

# State Variables

# Water stored in the upper reservoir in hydropower generation point at the end of time t
SU_t = p_model.addVars(range(0, time), lb = 0, name="SU_t")

# water stored in the lower reservoir at point i at the end of time t
SL_t = p_model.addVars(range(0, time), lb = 0, name="SL_t")

# mismatched demand at the end of time t
Z_t = p_model.addVars(range(0, time), lb = 0, name="Z_t")

# Electricity sent from hydropower generation point to demand point at the end of time t
Tsd_t = p_model.addVars(range(0, time), lb = 0, name="Tsd_t")

# Electricity sent from demand point to hydropower generation at the end of time t
Tds_t = p_model.addVars(range(0, time), lb = 0, name="Tds_t")

# Water spilled from upper reservoir at point i at the end of time t
LU_t = p_model.addVars(range(0, time), lb = 0, name="LU_t")

# Water spilled from lower reservoir at point i at the end of time t
LL_t = p_model.addVars(range(0, time), lb = 0, name="LL_t")

# Solar energy internally used in point j at the end of time t
V_t = p_model.addVars(range(0, time), lb = 0, name="V_t")

# Water released from upper reservoir at point i at the end of time t
R_t = p_model.addVars(range(0, time), lb = 0, name="R_t")

# Water pumped from lower reservoir at point i at the end of time t
P_t = p_model.addVars(range(0, time), lb = 0, name="P_t")

# Upper reservoir capacity 
SUMax = p_model.addVar(lb=0, name="SUMax") 
# Removed vtype=grb.GRB.CONTINUOUS: By default, Gurobi assumes variables to be continuous, so you don't need to specify the variable type explicitly.

# Lower reservoir capacity 
SLMax = p_model.addVar(lb = 0, name="SLMax")

# Solar panel size 
SS = p_model.addVar(lb = 0, name="SS")

# Generator size 
PGMax = p_model.addVar(lb = 0, name="PGMax")

# Maximum energy (capacity) transmitted from generator to demand points
TMax = p_model.addVar(lb = 0, name="TMax")

# Ip - binary variable to prevent pump and release&spill happening at the same time. 
Ip = p_model.addVars(range(0, time), vtype=grb.GRB.BINARY, name="Ip")

# Isd - binary variable to prevent sending electricity to and from village
Isd = p_model.addVars(range(0, time), vtype=grb.GRB.BINARY, name="Isd")

# CONSTRAINTS

#(0)
#p_model.addConstr((SLMax == 0))

#(1)
p_model.addConstrs((SL_t[t] <= SLMax)  for t in range(0, time))

#(2)
p_model.addConstrs((SU_t[t] <= SU_t[t-1]  + inflow[t] + P_t[t] - R_t[t] - LU_t[t]) for t in range(1, time))

#(3)
p_model.addConstr(SU_t[0] <=  SUMax + inflow[0] + P_t[0] - R_t[0] - LU_t[0])

#(4) # paperda bu yok!!!!!
p_model.addConstrs((SU_t[t] <= SUMax) for t in range(0, time))

#(5)
p_model.addConstrs((SL_t[t] == SL_t[t-1] + R_t[t] - P_t[t] - LL_t[t]) for t in range(1, time))

#(6)
p_model.addConstr(SL_t[0] <=  R_t[0] - P_t[0] - LL_t[0])

#(7) 
p_model.addConstrs((R_t[t]*d*g*h*alpha/3600000 <= PGMax*n) for t in range(0, time))

#(8)
p_model.addConstrs((P_t[t]*d*g*h/alpha/3600000 <= PGMax*n) for t in range(0, time))

#(9)
p_model.addConstrs((Tsd_t[t] == R_t[t]*d*g*h*alpha/3600000) for t in range(0, time))

#(10)
p_model.addConstrs((P_t[t]*d*g*h/alpha/3600000 == Tds_t[t]*(1-l)) for t in range(0, time))

#(11)
p_model.addConstrs((Tsd_t[t] <= TMax*n) for t in range(0, time))

#(12)
p_model.addConstrs((Tds_t[t] <= TMax*n) for t in range(0, time))

#(13)
p_model.addConstrs((solar_radiation[t]*SS*gamma*n >= V_t[t] + Tds_t[t])  for t in range(0, time))

#(14)
p_model.addConstrs((Z_t[t] == demand[t] - V_t[t] - Tsd_t[t]*(1-l)) for t in range(0, time))

#(15)
p_model.addConstrs((P_t[t] <= Ip[t]*M) for t in range(0, time))

#(16)
p_model.addConstrs((R_t[t] <= (1-Ip[t])*M) for t in range(0, time))

#(17)
p_model.addConstrs((LU_t[t] <= (1-Ip[t])*M) for t in range(0, time))

#(18)
p_model.addConstrs((Tsd_t[t] <= Isd[t]*M) for t in range(0, time))

#(19)
p_model.addConstrs((Tds_t[t] <= (1-Isd[t])*M) for t in range(0, time))

#(20)
p_model.addConstr(SU_t[time-1] == SUMax)

Objective = d_h*C_s*(SUMax+SLMax)  + d_h*C_pg*PGMax + d_s*SS*C_m + d_t*C_t*TMax + quicksum(Z_t[t]*mu*n for t in range(0, time))


p_model.setObjective(Objective, GRB.MINIMIZE)

p_model.optimize()

#%%
# OUTPUTS

# The objective value
print("The objective value is $", round(c_model.objVal,2), ". (Conventional)")

# Upper reservoir capacity 
print("For Conventional", 
      "\nSUMax = ", c_model.getVarByName("SUMax").X, "m3",
      "\nSLMax = ", c_model.getVarByName("SLMax").X, "m3",
      "\nSS = ", c_model.getVarByName("SS").X, "m2",
      "\nPGMax = ", c_model.getVarByName("PGMax").X, "kW",
      "\nTMax = ", c_model.getVarByName("TMax").X, "kW")

names = ["SU_t", "SL_t", "Z_t", "Tsd_t", "Tds_t", "LU_t", "LL_t", "V_t", "R_t", "P_t"]
df_c = pd.DataFrame(columns=names)

for t in range(0, time):  # Assuming 2920 time steps
    for name in names:
        variable_name = f'{name}[{t}]'
        variable_value = c_model.getVarByName(variable_name).X
        df_c.loc[t, name] = variable_value

df_c["demand"] = demand
df_c["SUMax"] = c_model.getVarByName("SUMax").X
df_c["SLMax"] = c_model.getVarByName("SLMax").X
df_c["SS"] = c_model.getVarByName("SS").X
df_c["PGMax"] = c_model.getVarByName("PGMax").X
df_c["TMax"] = c_model.getVarByName("TMax").X

#df.to_csv("C:/Users/beyza/Desktop/output.csv", sep='\t')

# pumped

# The objective value
print("The objective value is $", round(p_model.objVal,2), ". (PHES)")

# Upper reservoir capacity 
print("For PHES", 
      "\nSUMax = ", p_model.getVarByName("SUMax").X, "m3",
      "\nSLMax = ", p_model.getVarByName("SLMax").X, "m3",
      "\nSS = ", p_model.getVarByName("SS").X, "m2",
      "\nPGMax = ", p_model.getVarByName("PGMax").X, "kW",
      "\nTMax = ", p_model.getVarByName("TMax").X, "kW")

names = ["SU_t", "SL_t", "Z_t", "Tsd_t", "Tds_t", "LU_t", "LL_t", "V_t", "R_t", "P_t"]
df_p = pd.DataFrame(columns=names)

for t in range(0, time):  # Assuming 2920 time steps
    for name in names:
        variable_name = f'{name}[{t}]'
        variable_value = p_model.getVarByName(variable_name).X
        df_p.loc[t, name] = variable_value

df_p["demand"] = demand
df_p["SUMax"] = p_model.getVarByName("SUMax").X
df_p["SLMax"] = p_model.getVarByName("SLMax").X
df_p["SS"] = p_model.getVarByName("SS").X
df_p["PGMax"] = p_model.getVarByName("PGMax").X
df_p["TMax"] = p_model.getVarByName("TMax").X




#%% GRAPHS

#%% Figure 5
# Water stored in the upper and lower reservoirs

month_interval = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 245, 245])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create subplots with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot the first line plot
ax1.plot(range(0, time), df_p["SU_t"]/10**9, label='Upper reservoir')
ax1.set_ylabel('$km^3$')
ax1.set_title('Water Stored in the Upper & Lower Reservoirs')

# Plot the second line plot
ax2.plot(range(0, time), df_p["SL_t"]/10**9, label='Lower reservoir')
ax2.set_ylabel('$km^3$')

# Set x-axis tick labels to months
ax2.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax2.set_xticklabels(months)

ax1.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax1.set_xticklabels(months)

# Add legends to the plots
ax1.legend(loc="upper right", fontsize="small")
ax2.legend(loc="upper right", fontsize="small")

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Display the plots
plt.show()


#%% Figure 6

month_interval = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 245, 245])
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create subplots with 2 rows and 1 column
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

# Plot the first line plot
ax1.plot(range(0, time), inflow/10**9, color= "red",label='Inflow')
ax1.set_ylabel('$km^3$')

# Plot the second line plot
ax2.plot(range(0, time), df_p["R_t"]/10**9, color= "purple", label='Release')
ax2.set_ylabel('$km^3$')

# Plot the third line plot
ax3.plot(range(0, time), df_p["P_t"]/10**9, color= "orange", label='Pump')
ax3.set_ylabel('$km^3$')

# Plot the fourth line plot
ax4.plot(range(0, time), df_p["LL_t"]/10**9, color= "blue", label='Spill')
ax4.set_ylabel('$km^3$')

# Set x-axis tick labels to months
ax1.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax1.set_xticklabels(months)

ax2.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax2.set_xticklabels(months)

ax3.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax3.set_xticklabels(months)

ax4.set_xticks(np.cumsum(month_interval) - month_interval[0])
ax4.set_xticklabels(months)


# Add legends to the plots
ax1.legend(loc="upper right", fontsize="small")
ax2.legend(loc="upper right", fontsize="small")
ax3.legend(loc="upper right", fontsize="small")
ax4.legend(loc="upper right", fontsize="small")

# close scientific notation
ax1.ticklabel_format(style='plain', axis='y')
ax2.ticklabel_format(style='plain', axis='y')
ax3.ticklabel_format(style='plain', axis='y')
ax4.ticklabel_format(style='plain', axis='y')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Display the plots
plt.show()

#%% Figure 7

# First week of June
plt.clf()

day_interval = np.array([8, 8, 8, 8, 8, 8, 8])
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']

# Create subplots with 2 rows and 1 column
fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(range(0, 56), inflow[1212:1268]/10**9, color= "red",label='Inflow')
ax1.plot(range(0, 56), df_p.iloc[1212:1268]["R_t"]/10**9, color= "purple", label='Release')
ax1.plot(range(0, 56), df_p.iloc[1212:1268]["P_t"]/10**9, color= "orange", label='Pump')
ax1.plot(range(0, 56), df_p.iloc[1212:1268]["LU_t"]/10**9, color= "blue", label='Spill')
ax1.set_ylabel('$km^3$')


ax2.plot(range(0, 56), demand[1212:1268]/10**6, color= "red",label='Demand')
ax2.plot(range(0, 56), solar_radiation[1212:1268]*p_model.getVarByName("SS").X*gamma*n/10**6, color= "purple", label='Solar')
ax2.plot(range(0, 56), df_p.iloc[1212:1268]["Tsd_t"]/10**6, color= "orange", label='Hydro')
ax2.plot(range(0, 56), df_p.iloc[1212:1268]["Z_t"]/10**6, color= "blue", label='Diesel')
ax2.set_ylabel('GWh')

# Add legends to the plots
ax1.legend(loc="upper right", fontsize="small", ncol=4)
ax2.legend(loc="upper right", fontsize="small", ncol=4)


# Set x-axis tick labels to months
ax2.set_xticks(np.cumsum(day_interval) - day_interval[0])
ax2.set_xticklabels(days)

ax1.set_xticks(np.cumsum(day_interval) - day_interval[0])
ax1.set_xticklabels(days)


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Display the plots
plt.show()

#%%

plt.clf()
# FIGURE 8
categories = ["Hydro", "Solar", "Diesel"]

hydro_used = df_c["Tsd_t"].sum()
solar_v = df_c["V_t"].sum()
solar_tds = df_c["Tds_t"].sum()
diesel_used = df_c["Z_t"].sum()
total = hydro_used + solar_v + solar_tds + diesel_used
production_c = [hydro_used*100/total, (solar_v + solar_tds)*100/total, diesel_used*100/total]

hydro_used = df_p["Tsd_t"].sum()
solar_used = df_p["V_t"].sum() + df_p["Tds_t"].sum()
diesel_used = df_p["Z_t"].sum()
total = hydro_used + solar_used + diesel_used
production_p = [hydro_used*100/total, solar_used*100/total, diesel_used*100/total]

# Create subplots with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot the first bar plot
ax1.bar(categories, production_c)
ax1.bar(categories[1], solar_tds*100/total, bottom=solar_v*100/total, hatch='//')  # Add dash lines to the second part
ax1.set_xlabel('Energy Sources')
ax1.set_ylabel('Resource Distribution (%)')
ax1.set_title('Conventional')

# Plot the second bar plot
ax2.bar(categories, production_p)
ax2.bar(categories[1], df_p["Tds_t"].sum()*100/total, bottom=df_p["V_t"].sum()*100/total,fill=False, hatch='//', edgecolor='black')
ax2.set_xlabel('Energy Sources')
ax2.set_ylabel('Resource Distribution (%)')
ax2.set_title('PHES')

# Create a custom legend entry for the dashed part
legend_entry = mpatches.Patch(fill=False, hatch='//', edgecolor='black', label='Pump')

# Add the legend to the plots
ax2.legend(handles=[legend_entry], loc='upper right')
# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

plt.show()

#%% Figure 9

# Create subplots with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2)

# Set the x-axis values
x = range(1,9)

# Define the bar width
bar_width = 0.3

# Plot the bar plot
ax1.bar(x, solar_radiation[4:12]*p_model.getVarByName("SS").X*gamma*n/10**6, width=bar_width, color='grey', label='Spilled')
ax1.bar(x, df_p.iloc[4:12]["V_t"]/10**6, width=bar_width, bottom=solar_radiation[4:12]*p_model.getVarByName("SS").X*gamma*n/10**6, color='orange', label='Internal')
ax1.bar(x, df_p.iloc[4:12]["Tds_t"]/10**6, width=bar_width, bottom=(solar_radiation[4:12]*p_model.getVarByName("SS").X*gamma*n + df_p.iloc[4:12]["V_t"])/10**6, color='blue', label='Pumped')
ax1.set_xlabel('Time (3hr)')
ax1.set_ylabel('GWh')
ax1.set_title('Pumped Hydro System')
ax1.set_xticks(x)
ax1.set_xticklabels(x)

# Plot the second bar plot
ax2.bar(x, solar_radiation[4:12]*c_model.getVarByName("SS").X*gamma*n/10**6, width=bar_width, color='grey', label='Spilled')
ax2.bar(x, df_c.iloc[4:12]["V_t"]/10**6, width=bar_width, bottom=solar_radiation[4:12]*c_model.getVarByName("SS").X*gamma*n/10**6, color='orange', label='Internal')
ax2.bar(x, df_c.iloc[4:12]["Tds_t"]/10**6, width=bar_width, bottom=(solar_radiation[4:12]*c_model.getVarByName("SS").X*gamma*n + df_c.iloc[4:12]["V_t"])/10**6, color='blue', label='Pumped')
ax2.set_xlabel('Time (3hr)')
ax2.set_ylabel('GWh')
ax2.set_title('Conventional System')
ax2.set_xticks(x)
ax2.set_xticklabels(x)

# Add legends to the plots
ax1.legend(loc="upper right", fontsize="small")
ax2.legend(loc="upper right", fontsize="small")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Display the plots
plt.show()




# %% Our plots
# plot solar radiation
plt.clf()
monthly = []

month_interval = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 245, 245])

cumsum_int = np.cumsum(month_interval)

for i in range(0, len(cumsum_int)):
        if i == 0:
            monthly.append(sum(solar_radiation[:cumsum_int[i]]))
        if i != 0:
            monthly.append(sum(solar_radiation[cumsum_int[i-1]:cumsum_int[i]]))

monthly = np.array(monthly)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.bar(months, monthly)
plt.xlabel('Months')
plt.ylabel('kWh/m2')
plt.title('Total Solar Radiation per Month')
plt.xticks(range(len(months)), months, rotation='vertical')
plt.show()



# %%
# plot inflow 
plt.clf()
monthly = []

month_interval = np.array([243, 243, 243, 243, 243, 243, 243, 243, 243, 243, 245, 245])

cumsum_int = np.cumsum(month_interval)

for i in range(0, len(cumsum_int)):
        if i == 0:
            monthly.append(sum(inflow[:cumsum_int[i]]))
        if i != 0:
            monthly.append(sum(inflow[cumsum_int[i-1]:cumsum_int[i]]))

monthly = np.array(monthly)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

plt.bar(months, monthly)
plt.xlabel('Months')
plt.ylabel('$m^3/3hr$')
plt.title('River Discharge')
plt.xticks(range(len(months)), months, rotation='vertical')
plt.show()
# %%
