from django.db import models

# Create your models here.

class Train(models.Model):
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

    class Meta:
        managed = False
        db_table = 'train'