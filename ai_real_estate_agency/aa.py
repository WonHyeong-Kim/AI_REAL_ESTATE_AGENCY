# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
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
    transaction_id = models.BigIntegerField(primary_key=True)
    apartment_id = models.BigIntegerField(blank=True, null=True)
    gu = models.BigIntegerField(blank=True, null=True)
    exclusive_use_area = models.FloatField(blank=True, null=True)
    year_of_completion = models.BigIntegerField(blank=True, null=True)
    transaction_year_month = models.BigIntegerField(blank=True, null=True)
    transaction_date = models.BigIntegerField(blank=True, null=True)
    floor = models.BigIntegerField(blank=True, null=True)
    park_area_sum = models.FloatField(blank=True, null=True)
    day_care_babyteacher_rate = models.FloatField(db_column='day_care_babyTeacher_rate', blank=True, null=True)  # Field name made lowercase.
    transaction_real_price = models.BigIntegerField(blank=True, null=True)
    cctv_num = models.BigIntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'train'
