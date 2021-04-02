from django.db import models

# Create your models here.

class DayCareCenter(models.Model):
    city = models.CharField(max_length=45, blank=True, null=True)
    gu = models.CharField(max_length=45, blank=True, null=True)
    day_care_name = models.CharField(max_length=100, blank=True, null=True)
    day_care_type = models.CharField(max_length=100, blank=True, null=True)
    day_care_baby_num = models.IntegerField(blank=True, null=True)
    teacher_num = models.IntegerField(blank=True, null=True)
    nursing_room_num = models.IntegerField(blank=True, null=True)
    playground_num = models.IntegerField(blank=True, null=True)
    cctv_num = models.IntegerField(db_column='CCTV_num', blank=True, null=True)  # Field name made lowercase.
    is_commuting_vehicle = models.CharField(max_length=20, blank=True, null=True)
    reference_date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'day_care_center'


class Park(models.Model):
    city = models.CharField(max_length=45, blank=True, null=True)
    gu = models.CharField(max_length=45, blank=True, null=True)
    dong = models.CharField(max_length=45, blank=True, null=True)
    park_name = models.CharField(max_length=100, blank=True, null=True)
    park_type = models.CharField(max_length=100, blank=True, null=True)
    park_area = models.IntegerField(blank=True, null=True)
    park_exercise_facility = models.CharField(max_length=200, blank=True, null=True)
    park_entertainment_facility = models.CharField(max_length=200, blank=True, null=True)
    park_benefit_facility = models.CharField(max_length=200, blank=True, null=True)
    park_cultural_facitiy = models.CharField(max_length=200, blank=True, null=True)
    park_facility_other = models.CharField(max_length=200, blank=True, null=True)
    park_open_year = models.TextField(blank=True, null=True)  # This field type is a guess.
    reference_date = models.DateField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'park'


class Submission(models.Model):
    transaction_id = models.AutoField(primary_key=True)
    transaction_real_price = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'submission'


class Test(models.Model):
    transaction = models.OneToOneField(Submission, models.DO_NOTHING, primary_key=True)
    apartment_id = models.IntegerField(blank=True, null=True)
    city = models.CharField(max_length=45, blank=True, null=True)
    dong = models.CharField(max_length=45, blank=True, null=True)
    jibun = models.CharField(max_length=45, blank=True, null=True)
    apt = models.CharField(max_length=250, blank=True, null=True)
    addr_kr = models.CharField(max_length=250, blank=True, null=True)
    exclusive_use_area = models.FloatField(blank=True, null=True)
    year_of_completion = models.TextField(blank=True, null=True)  # This field type is a guess.
    transaction_year_month = models.DateField(blank=True, null=True)
    transaction_date = models.CharField(max_length=45, blank=True, null=True)
    floor = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'test'


class Train(models.Model):
    transaction_id = models.AutoField(primary_key=True)
    apartment_id = models.IntegerField(blank=True, null=True)
    city = models.CharField(max_length=45, blank=True, null=True)
    dong = models.CharField(max_length=45, blank=True, null=True)
    jibun = models.CharField(max_length=45, blank=True, null=True)
    apt = models.CharField(max_length=45, blank=True, null=True)
    addr_kr = models.CharField(max_length=45, blank=True, null=True)
    exclusive_use_area = models.FloatField(blank=True, null=True)
    year_of_completion = models.TextField(blank=True, null=True)  # This field type is a guess.
    transaction_year_month = models.DateField(blank=True, null=True)
    transaction_date = models.CharField(max_length=45, blank=True, null=True)
    floor = models.IntegerField(blank=True, null=True)
    transaction_real_price = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'train'