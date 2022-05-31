#preg	plas	pres	skin	insulin	mass	pedi	age	result
#import libraries
#conda install scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#app heading

#A streamlit app with two centered texts with different seizes
import streamlit as st


st.image('logo2.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st.markdown("<h3 style='text-align: center; color: grey;'>The Nelson Mandela African Institution of Science and Technology (NM-AIST)</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: black;'>Machine Learning Model for Predicting Construction Project Success in Tanzania</h3>", unsafe_allow_html=True)



#creating sidebar for user input features
st.sidebar.header('User Input Parameters')

 #st.number_input(label, min_value=None, max_value=None, value=, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None) 



def user_input_features():
        
        Completiontime = st.sidebar.number_input('Completiontime', 0, 6)
        ConstructionCost = st.sidebar.number_input('ConstructionCost', 0, 6)
        QualityofaConstructionProject = st.sidebar.number_input('QualityofaConstructionProject', 0, 6)
        Communicationbetweenprojectstakeholders = st.sidebar.number_input('Communicationbetweenprojectstakeholders ', 0, 6)
        TopManagementsupport=st.sidebar.number_input('TopManagementsupport', 0, 6)
        ProjectInitialPlanning=st.sidebar.number_input('ProjectInitialPlanning', 0, 6)
        Projectrequirementsandobjectives=st.sidebar.number_input('Projectrequirementsandobjectives', 0, 6 )
        Projectreportinginprojectlifecycle=st.sidebar.number_input('Projectreportinginprojectlifecycle', 0, 6 )
        ProjectManagercompetence=st.sidebar.number_input('ProjectManagercompetence', 0, 6 )
        Projecttechnicalteamexperienceandcompetence=st.sidebar.number_input('Projecttechnicalteamexperienceandcompetence', 0, 6 )
        Changemanagementprocess=st.sidebar.number_input('Changemanagementprocess', 0, 6 )
        Projectdesiredoutcome=st.sidebar.number_input('Projectdesiredoutcome', 0, 6 )
        Userclientsatisfaction=st.sidebar.number_input('Userclientsatisfaction', 0, 6 )
        Projectteamsatisfaction=st.sidebar.number_input('Projectteamsatisfaction', 0, 6 )
        TopManagementsatisfaction=st.sidebar.number_input('TopManagementsatisfaction', 0, 6 )
        Clientfinancialcapacity=st.sidebar.number_input('Clientfinancialcapacity', 0, 6 )
        Projectteamcommitment=st.sidebar.number_input('Projectteamcommitment', 0, 6 )
        ProjectScope=st.sidebar.number_input('ProjectScope', 0, 6 )
        ProjectRiskManagement=st.sidebar.number_input('ProjectRiskManagement', 0, 6 )
        Politicalenvironment=st.sidebar.number_input('Politicalenvironment', 0, 6 )
        Competenceinprojectmanagement=st.sidebar.number_input('Competenceinprojectmanagement', 0, 6 )
        Functionalrequirements=st.sidebar.number_input('Functionalrequirements', 0, 6 )
        Monitoringandcontrol=st.sidebar.number_input('Monitoringandcontrol', 0, 6 )
        EffectiveAntiCorruptionpolicy=st.sidebar.number_input('EffectiveAntiCorruptionpolicy', 0, 6 )
        Projectteamsize=st.sidebar.number_input('Projectteamsize', 0, 6 )
        Projectteammotivation=st.sidebar.number_input('Projectteammotivation', 0, 6 )
        Projectmanagementmethodologies=st.sidebar.number_input('Projectmanagementmethodologies', 0, 6 )
        Environmentalimpact=st.sidebar.number_input('Environmentalimpact', 0, 6 )
        Socialeconomicenvironment=st.sidebar.number_input('Socialeconomicenvironment', 0, 6 )
        Technologicalenvironment=st.sidebar.number_input('Technologicalenvironment', 0, 6 )
        Sitelimitationandlocation=st.sidebar.number_input('Sitelimitationandlocation', 0, 6 )
        Constructability=st.sidebar.number_input('Constructability', 0, 6 )
        Formaldisputeresolutionprocess=st.sidebar.number_input('Formaldisputeresolutionprocess', 0, 6 )
        Consultantcompetence=st.sidebar.number_input('Consultantcompetence', 0, 6 )
        Levelofautomation=st.sidebar.number_input('Levelofautomation', 0, 6 )
        Levelofskilledlaborrequired=st.sidebar.number_input('Levelofskilledlaborrequired', 0, 6 )
        
        
        SiteInspections=st.sidebar.number_input('SiteInspections', 0, 6 )
        Projectvisionandmission=st.sidebar.number_input('Projectvisionandmission', 0, 6 )
        QualityAssurance=st.sidebar.number_input('QualityAssurance', 0, 6 )
        Realtimeprojectauditing=st.sidebar.number_input('Realtimeprojectauditing', 0, 6 )
        Contractmanagementandperformance=st.sidebar.number_input('Contractmanagementandperformance', 0, 6 )


        
        data = {
                 'Constructability':Constructability,
                 'Competenceinprojectmanagement':Competenceinprojectmanagement,
                 'Monitoringandcontrol': Monitoringandcontrol,
                 'EffectiveAntiCorruptionpolicy': EffectiveAntiCorruptionpolicy,
                 'Socialeconomicenvironment': Socialeconomicenvironment,
                 'Projectdesiredoutcome':Projectdesiredoutcome,
                 'Communicationbetweenprojectstakeholders':Communicationbetweenprojectstakeholders,
                 'SiteInspections':SiteInspections,
                 'Projectvisionandmission':Projectvisionandmission,
                 'ProjectScope': ProjectScope,
                 'Projectteamsize': Projectteamsize,
                 'Projectteamsatisfaction': Projectteamsatisfaction,
                 'Projectreportinginprojectlifecycle': Projectreportinginprojectlifecycle,
                 'Realtimeprojectauditing' : Realtimeprojectauditing,
                 'Projectrequirementsandobjectives': Projectrequirementsandobjectives,
                 'Levelofskilledlaborrequired': Levelofskilledlaborrequired,
                 'TopManagementsatisfaction': TopManagementsatisfaction,
                 'ProjectRiskManagement':ProjectRiskManagement,
                 'Environmentalimpact': Environmentalimpact,
















                

      
               }
        features = pd.DataFrame(data, index=[0])
        
        
        
        return features
df = user_input_features()



#st.table(df)

#st.subheader('User Input parameters')
#st.write(df)
#reading csv file
data=pd.read_csv("./1k_aug.csv")
#X = data[['Completiontime',	'ConstructionCost',	'QualityofaConstructionProject',	'Communicationbetweenprojectstakeholders',	'TopManagementsupport',	'ProjectInitialPlanning',	'Projectrequirementsandobjectives',	'Projectreportinginprojectlifecycle',	'ProjectManagercompetence',	'Projecttechnicalteamexperienceandcompetence'	,'Changemanagementprocess',	'Projectdesiredoutcome',	'Userclientsatisfaction',	'Projectteamsatisfaction',	'TopManagementsatisfaction',	'Clientfinancialcapacity',	'Projectteamcommitment',	'ProjectScope',	'ProjectRiskManagement',	'Politicalenvironment',	'Competenceinprojectmanagement',	'Functionalrequirements',	'Monitoringandcontrol',	'EffectiveAntiCorruptionpolicy',	'Projectteamsize',	'Projectteammotivation'	,'Projectmanagementmethodologies'	,'Environmentalimpact'	,'Socialeconomicenvironment'	,'Technologicalenvironment'	,'Sitelimitationandlocation'	,'Constructability',	'Formaldisputeresolutionprocess',	'Consultantcompetence',	'Levelofautomation',	'Levelofskilledlaborrequired',	'SiteInspections',	'Projectvisionandmission',	'QualityAssurance',	'Realtimeprojectauditing']]


X =np.array(data[['Constructability', 'Competenceinprojectmanagement',  'Monitoringandcontrol', 
 'EffectiveAntiCorruptionpolicy',  'Socialeconomicenvironment',
'Projectdesiredoutcome',
 'Communicationbetweenprojectstakeholders',
 'SiteInspections',  'Projectvisionandmission',
 'ProjectScope', 'Projectteamsize',  'Projectteamsatisfaction', 
  'Projectreportinginprojectlifecycle',  'Realtimeprojectauditing', 
  'Projectrequirementsandobjectives', 'Levelofskilledlaborrequired',
 'TopManagementsatisfaction',  'ProjectRiskManagement', 
  'Environmentalimpact']])
Y = np.array(data['Successifulness'])
#random forest model
rfc= RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=None, min_samples_split=2, 
                           min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True, 
                           oob_score=False, n_jobs=1, random_state=42, verbose=0)
rfc.fit(X, Y)
#st.caption('Contract Management and Performance, 1 = *Highly Likely to Succeed*, 0 = *Not Likely to Succeed*')
st.write(pd.DataFrame({
  'SCORE': ["Not Likely to Succeed","Highly Likely to Succeed"]}))

prediction = rfc.predict(df)
prediction_proba = rfc.predict_proba(df)
#st.subheader('Prediction')
#st.write(prediction)

#st.subheader('Prediction Probability')

#st.write(prediction_proba)

from sklearn.model_selection import train_test_split
X = data[['Constructability', 'Competenceinprojectmanagement',  'Monitoringandcontrol', 
 'EffectiveAntiCorruptionpolicy',  'Socialeconomicenvironment',
'Projectdesiredoutcome',
 'Communicationbetweenprojectstakeholders',
 'SiteInspections',  'Projectvisionandmission',
 'ProjectScope', 'Projectteamsize',  'Projectteamsatisfaction', 
  'Projectreportinginprojectlifecycle',  'Realtimeprojectauditing', 
  'Projectrequirementsandobjectives', 'Levelofskilledlaborrequired',
 'TopManagementsatisfaction',  'ProjectRiskManagement', 
  'Environmentalimpact']]
y = data.Successifulness
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.30, random_state=0
 )

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rnd_clf.fit(X_train, y_train)

prediction = rnd_clf.predict(X_test)

from sklearn.metrics import accuracy_score


prediction = rnd_clf.predict(df)
prediction_proba = rnd_clf.predict_proba(df)
st.subheader('Prediction Probability')
st.write(prediction_proba)