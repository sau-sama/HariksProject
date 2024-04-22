from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)


class Loan_Status_Prediction(models.Model):
    Idno= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Experience= models.CharField(max_length=300)
    Income= models.CharField(max_length=300)
    ZIP_Code= models.CharField(max_length=300)
    Family= models.CharField(max_length=300)
    CCAvg= models.CharField(max_length=300)
    Education= models.CharField(max_length=300)
    Mortgage= models.CharField(max_length=300)
    Securities_Account= models.CharField(max_length=300)
    CD_Account= models.CharField(max_length=300)
    Online= models.CharField(max_length=300)
    CreditCard= models.CharField(max_length=300)
    prediction_svm= models.CharField(max_length=300)
    prediction_Logistic= models.CharField(max_length=300)
    prediction_rfc= models.CharField(max_length=300)
    prediction_dtc= models.CharField(max_length=300)
    prediction_knc= models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)


