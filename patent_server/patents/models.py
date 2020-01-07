from django.db import models


# Create your models here.

class PatentDes(models.Model):
    number = models.CharField(max_length=20)
    title = models.CharField(max_length=1000)
    technical_field = models.CharField(max_length=1000)
    background_art = models.CharField(max_length=1000)
    invention_content = models.CharField(max_length=1000)
    drawings = models.CharField(max_length=1000)
    implementation = models.CharField(max_length=1000)

class PatentClaim(models.Model):
    pass
