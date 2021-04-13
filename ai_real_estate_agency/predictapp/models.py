# Create your models here.
from django.db import models

class Dataset(models.Model):
    transaction_id = models.BigIntegerField(primary_key=True)
    apartment_id = models.BigIntegerField(blank=True, null=True)
    city = models.TextField(blank=True, null=True)
    dong = models.TextField(blank=True, null=True)
    jibun = models.TextField(blank=True, null=True)
    apt = models.TextField(blank=True, null=True)
    addr_kr = models.TextField(blank=True, null=True)
    exclusive_use_area = models.FloatField(blank=True, null=True)
    year_of_completion = models.BigIntegerField(blank=True, null=True)
    transaction_year_month = models.BigIntegerField(blank=True, null=True)
    transaction_date = models.TextField(blank=True, null=True)
    floor = models.BigIntegerField(blank=True, null=True)
    transaction_real_price = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dataset'


class Gu(models.Model):
    gu_name = models.CharField(primary_key=True, max_length=30)
    gu_num = models.BigIntegerField(blank=True, null=True)
    gu_area = models.FloatField(blank=True, null=True)
    gu_daycare = models.FloatField(blank=True, null=True)
    gu_cctv = models.BigIntegerField(blank=True, null=True)
    gu_mean_price = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'gu'


class Train(models.Model):
<<<<<<< HEAD
    apartment_id = models.BigIntegerField(blank=True, primary_key = True)
=======
    transaction_id = models.BigIntegerField(primary_key=True)
    apartment_id = models.BigIntegerField(blank=True, null=True)
>>>>>>> 01ea815644f4b3c2e0ab38449a602602c6e2c11e
    gu = models.BigIntegerField(blank=True, null=True)
    exclusive_use_area = models.FloatField(blank=True, null=True)
    year_of_completion = models.BigIntegerField(blank=True, null=True)
    transaction_year_month = models.BigIntegerField(blank=True, null=True)
    transaction_date = models.BigIntegerField(blank=True, null=True)
    floor = models.BigIntegerField(blank=True, null=True)
    park_area_sum = models.FloatField(blank=True, null=True)
    day_care_babyteacher_rate = models.FloatField(db_column='day_care_babyTeacher_rate', blank=True, null=True)  # Field name made lowercase.
    transaction_real_price = models.BigIntegerField(blank=True, null=True)
    number_of_cctv = models.BigIntegerField(db_column='number of cctv', blank=True, null=True)  # Field renamed to remove unsuitable characters.

    class Meta:
        managed = False
        db_table = 'train'
