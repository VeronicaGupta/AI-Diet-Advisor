import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import random
from pprint import pprint 
import sys
import json

# parse food database
data = pd.read_csv('food.csv')

# parse nutritional distribution (trained data)
datafin = pd.read_csv('nutrition_distriution.csv')

# define meal data from food DB
Food_itemsdata=data['Food_items']
Breakfastdata=data['Breakfast']
Lunchdata=data['Lunch']
Dinnerdata=data['Dinner']
VegNovVeg=data['VegNovVeg']

# define BMI class and age class
categories = {
    (0, 16): "Severely Underweight",
    (16, 18.5): "Underweight",
    (18.5, 25): "Healthy",
    (25, 30): "Overweight",
    (30, float("inf")): "Severely Overweight"
}
# 0Calories ,1Fats (gm),2Proteins(g),3Iron(mg),4Calcium(mg),5Sodium(mg),6Potassium(mg),7Carbohydrates (gm),8Fibre (gm),9Vitamin D (mcg),10Sugars (gm)
loss_param = [1,2,7,8]
gain_param = [0,1,2,3,4,7,9,10]
healthy_param = [1,2,3,4,5,6,7,8,9]

max_cluster_b = 4
max_cluster_l = 4
max_cluster_d = 4

init_b=1
init_l=1
init_d=1

state_b=0
state_l=0
state_d=0

meal_size = 2

def diet_planner():
    # INPUTS
    args = sys.argv
    age = int(args[1]) #int(input("Enter your age: "))
    veg = args[2] #float(input("Enter your vegetarianism score (0-1): "))
    weight = float(args[3]) #float(input("Enter your weight in kg: "))
    height = float(args[4]) #float(input("Enter your height in cm: "))
    goal = args[5] #1,-1,0
    gender = args[6] #1:f,0:m

    if veg == "Veg":
        veg = 1
    elif veg == "Non-Veg":
        veg = 0

    if goal == "Weight Loss":
        goal = -1
    elif goal == "Weight Gain":
        goal = 1
    elif goal == "Healthy":
        goal = 0

    if gender=="Male":
        gender=0
    elif gender=="Female":
        gender=1


    # get bmi class
    bmi = weight / ((height / 100) ** 2)
    for range_, category in categories.items():
        if range_[0] <= bmi < range_[1]:
            bmicls = len(categories) - list(categories.keys()).index(range_) - 1
            bmi_classes = category
            break
    
    # get ideal intake
    if gender==0:
        bmr_ideal = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    elif gender==1:
        bmr_ideal = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)


    # print(f" \
    #             AGE: {age:d}\n \
    #             VEG: {veg:d}\n \
    #             WEIGHT: {weight:f}kg\n \
    #             HEIGHT: {height:f}cm\n \
    #             GOAL: {goal:d}\n \
    #             GENDER: {gender:d}\n\n \
    #             BMI: {bmi:.2f}\n \
    #             BMI Class: {bmi_classes}\n \
    #             Age Class: {age_classes[agecls]}\n \
    #             Ideal Intake: {bmr_ideal:.2f}kCal\n ")
    
    # run ML

    if veg == 1:
        Breakfastfoodseparated = data.iloc[1:].loc[(Breakfastdata == 1) & (VegNovVeg == 1)].iloc[:, 0:]
        Lunchfoodseparated     = data.iloc[1:].loc[(Lunchdata == 1) & (VegNovVeg == 1)].iloc[:, 0:]
        Dinnerfoodseparated    = data.iloc[1:].loc[(Dinnerdata == 1) & (VegNovVeg == 1)].iloc[:, 0:]

        # clustering all 3 meals in 3 clusters 
        BreakfastfoodseparatedIDdata = data.iloc[1:].loc[(Breakfastdata == 1) & (VegNovVeg == 1)].iloc[:, 5:]
        LunchfoodseparatedIDdata     = data.iloc[1:].loc[(Lunchdata == 1) & (VegNovVeg == 1)].iloc[:, 5:]
        DinnerfoodseparatedIDdata    = data.iloc[1:].loc[(Dinnerdata == 1) & (VegNovVeg == 1)].iloc[:, 5:]
    else:
        Breakfastfoodseparated = data.iloc[1:].loc[Breakfastdata == 1].iloc[:, 0:]
        Lunchfoodseparated     = data.iloc[1:].loc[Lunchdata == 1].iloc[:, 0:]
        Dinnerfoodseparated    = data.iloc[1:].loc[Dinnerdata == 1].iloc[:, 0:]

        # clustering all 3 meals in 3 clusters 
        BreakfastfoodseparatedIDdata = data.iloc[1:].loc[Breakfastdata == 1].iloc[:, 5:]
        LunchfoodseparatedIDdata     = data.iloc[1:].loc[Lunchdata == 1].iloc[:, 5:]
        DinnerfoodseparatedIDdata    = data.iloc[1:].loc[Dinnerdata == 1].iloc[:, 5:]

    Datacalorie = BreakfastfoodseparatedIDdata.to_numpy(dtype='float')
    kmeans = KMeans(n_clusters=max_cluster_b, n_init=init_b, random_state=state_b).fit(Datacalorie)
    brklbl = kmeans.labels_

    Datacalorie = LunchfoodseparatedIDdata.to_numpy(dtype='float')
    kmeans  = KMeans(n_clusters=max_cluster_l, n_init=init_l, random_state=state_l).fit(Datacalorie)
    lnchlbl = kmeans.labels_

    Datacalorie = DinnerfoodseparatedIDdata.to_numpy(dtype='float')
    kmeans = KMeans(n_clusters=max_cluster_d, n_init=init_d, random_state=state_d).fit(Datacalorie)
    dnrlbl = kmeans.labels_

    # print(f"Breakfast: {brklbl}\nLunch: {lnchlbl}\nDinner: {dnrlbl}")


    # define nutritions required for loss, gain and healthy category or goal
    dataTog=datafin.T
    weightlosscat = dataTog.iloc[loss_param]
    weightgaincat = dataTog.iloc[gain_param]
    healthycat    = dataTog.iloc[healthy_param]

    # get data for goal categories
    weightlosscatDdata = weightlosscat.T.to_numpy()
    weightgaincatDdata = weightgaincat.T.to_numpy()
    healthycatDdata    = healthycat.T.to_numpy()

    weightlosscat = weightlosscatDdata[1:,0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:,0:len(weightgaincatDdata)]
    healthycat    = healthycatDdata[1:,0:len(healthycatDdata)]

    if goal==1:
        goal_cat = weightgaincat
        goal_param = gain_param
    elif goal==-1:
        goal_cat = weightlosscat
        goal_param = loss_param
    elif goal==0:
        goal_cat   = healthycat
        goal_param = healthy_param

    meals = {}
    meals["Breakfast"] = get_meals(goal_cat, brklbl,  bmr_ideal, len(goal_param), goal, Breakfastfoodseparated)
    meals["Lunch"] = get_meals(goal_cat, lnchlbl, bmr_ideal, len(goal_param), goal, Lunchfoodseparated)
    meals["Dinner"] = get_meals(goal_cat, dnrlbl,  bmr_ideal, len(goal_param), goal, Dinnerfoodseparated)

    # calculate target intake
    bmr=0
    calorie=0
    fat=0
    protien=0
    carbs=0
    fibre=0
    sugar=0

    calorie_d=0
    fat_d=0
    protien_d=0
    carbs_d=0
    fibre_d=0
    sugar_d=0

    # 0Calories ,1Fats (gm),2Proteins(g),3Iron(mg),4Calcium(mg),5Sodium(mg),6Potassium(mg),7Carbohydrates (gm),8Fibre (gm),9Vitamin D (mcg),10Sugars (gm)
    for _, i in meals.items():
        # print("\n", i["mealType"])
        calorie=0
        fat=0
        protien=0
        carbs=0
        fibre=0
        sugar=0
        for j in i:
            # print(j["foodName"])
            calorie+=float(j["calories"])
            fat+=9*float(j["fats"])
            protien+=4*float(j["protiens"])
            carbs+=4*float(j["carbohydrates"])
            fibre+=2*float(j["fibre"])
            sugar+=4*float(j["sugars"])

        calorie_d+=calorie
        fat_d+=fat
        protien_d+=protien
        carbs_d+=carbs
        fibre_d+=fibre
        sugar_d+=sugar

        # i["calorie"] = calorie
        # i["fat"] = fat
        # i["protien"] = protien
        # i["fibre"] = fibre
        # i["sugar"] = sugar
        # i.append(calorie)
        # i.append(fat)
        # i.append(protien)
        # i.append(fibre)
        # i.append(sugar)


    bmr=calorie_d+fat_d+protien_d+carbs_d+fibre_d+sugar_d
    
    # meals["daily_nutrition"] = {
    #                 "current_bmi":bmi_classes,
    #                 "ideal_bmr": bmr_ideal,
    #                 "target_bmr":bmr,
    #                 "calorie":calorie_d,
    #                 "fat" : fat_d,
    #                 "protien" : protien_d,
    #                 "fibre" : fibre_d,
    #                 "sugar" : sugar_d
    #                 }
               
    
    # OUTPUTS 

    # for i in meals:
    #     pprint(i) 
    print(json.dumps(meals)) 

    # print(f" \
    #         Target Intake: {bmr:.2f}kCal\n \
    #         Target Calorie: {calorie_d:.2f}kCal\n \
    #         Target Fat: {fat_d:.2f}kCal\n \
    #         Target Protien: {protien_d:.2f}kCal\n \
    #         Target Carbohydrates: {carbs_d:.2f}kCal\n \
    #         Target Fibre: {fibre_d:.2f}kCal\n \
    #         Target Sugar: {sugar_d:.2f}kCal\n")

    
    return meals
    

def select_kpi(arr, kpi):
    if np.count_nonzero(arr == kpi)>=3:
        return kpi
    else:
        arr = np.where(arr == kpi, 0, arr)
        new_max_kpi = np.amax(arr)
        return select_kpi(arr, new_max_kpi)

def get_meals(goal_cat, meal, bmr, param_size, goal, meal_data):
    if len(meal)>len(goal_cat):
        meal = meal[:len(goal_cat)]
    else:
        goal_cat = goal_cat[:len(meal)]

    for i in range(len(goal_cat)):
        valloc=list(goal_cat[i])

    X_test=np.zeros((len(goal_cat)*param_size, param_size),dtype=np.float32)
    for i in range(len(goal_cat)):
        valloc=list(goal_cat[i])
        X_test[i]=np.array(valloc)*bmr


    X_train = goal_cat # Features (trained variable)
    y_train = meal # Labels (target variable)

    clf=RandomForestClassifier(n_estimators=100)

    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    Y_pred= sum(np.array_split(y_pred, param_size))

    max_kpi = select_kpi(Y_pred, max(Y_pred))

    meal_food_data=meal_data["Food_items"].tolist()
    meal_calories=meal_data["Calories"].tolist()
    meal_fats=meal_data["Fats"].tolist()
    meal_protiens=meal_data["Proteins"].tolist()
    meal_iron=meal_data["Iron"].tolist()
    meal_calcium=meal_data["Calcium"].tolist()
    meal_sodium=meal_data["Sodium"].tolist()
    meal_potassium=meal_data["Potassium"].tolist()
    meal_carbohydrates=meal_data["Carbohydrates"].tolist()
    meal_fibre=meal_data["Fibre"].tolist()
    meal_vitamind=meal_data["VitaminD"].tolist()
    meal_sugars=meal_data["Sugars"].tolist()

    foods=[]
    #5-Calories, 6-Fats 6-Proteins, 7-Iron, 8-Calcium, 9-Sodium, 10-Potassium, 11-Carbohydrates, 12-Fibre, 13-VitaminD, 14-Sugars
    for i in range(len(Y_pred)):
        if Y_pred[i]==max_kpi:
            foods.append({
                "foodName": meal_food_data[i],
                "calories": meal_calories[i],
                "fats": meal_fats[i],
                "protiens": meal_protiens[i],
                "iron": meal_iron[i],
                "calcium": meal_calcium[i],
                "sodium": meal_sodium[i],
                "potassium": meal_potassium[i],
                "carbohydrates": meal_carbohydrates[i],
                "fibre": meal_fibre[i],
                "vitamind": meal_vitamind[i],
                "sugars": meal_sugars[i],
            })

    meal_intake = foods
    if len(foods)<meal_size:
        meal_intake = meal_data

    if len(foods)>=meal_size*2:
        if goal==1:
            foods = random.sample(meal_intake[(len(meal_intake)//2)-1:], meal_size)
        elif goal==-1:
            foods = random.sample(meal_intake[:len(meal_intake)//2], meal_size)
        elif goal==0:
            foods = random.sample(meal_intake[(len(meal_intake)//2)-meal_size:(len(meal_intake)//2)+meal_size], meal_size)
    else:
        foods = random.sample(meal_intake, meal_size)

    # print(X_train, len(X_train))
    # print(y_train, len(y_train))
    # print(y_pred, len(y_pred))
    # print(Y_pred, len(Y_pred), max_kpi)

    return foods
      

diet_planner()