# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class AuthGroup(models.Model):
    name = models.CharField(unique=True, max_length=150)

    class Meta:
        managed = False
        db_table = 'auth_group'


class AuthGroupPermissions(models.Model):
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)
    permission = models.ForeignKey('AuthPermission', models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_group_permissions'
        unique_together = (('group', 'permission'),)


class AuthPermission(models.Model):
    name = models.CharField(max_length=255)
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING)
    codename = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'auth_permission'
        unique_together = (('content_type', 'codename'),)


class AuthUser(models.Model):
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(unique=True, max_length=150)
    first_name = models.CharField(max_length=150)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'auth_user'


class AuthUserGroups(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    group = models.ForeignKey(AuthGroup, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_groups'
        unique_together = (('user', 'group'),)


class AuthUserUserPermissions(models.Model):
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)
    permission = models.ForeignKey(AuthPermission, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'auth_user_user_permissions'
        unique_together = (('user', 'permission'),)


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


class DjangoAdminLog(models.Model):
    action_time = models.DateTimeField()
    object_id = models.TextField(blank=True, null=True)
    object_repr = models.CharField(max_length=200)
    action_flag = models.PositiveSmallIntegerField()
    change_message = models.TextField()
    content_type = models.ForeignKey('DjangoContentType', models.DO_NOTHING, blank=True, null=True)
    user = models.ForeignKey(AuthUser, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'django_admin_log'


class DjangoContentType(models.Model):
    app_label = models.CharField(max_length=100)
    model = models.CharField(max_length=100)

    class Meta:
        managed = False
        db_table = 'django_content_type'
        unique_together = (('app_label', 'model'),)


class DjangoMigrations(models.Model):
    app = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    applied = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_migrations'


class DjangoSession(models.Model):
    session_key = models.CharField(primary_key=True, max_length=40)
    session_data = models.TextField()
    expire_date = models.DateTimeField()

    class Meta:
        managed = False
        db_table = 'django_session'


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


class News(models.Model):
    news_id = models.AutoField(primary_key=True)
    news_title = models.TextField(blank=True, null=True)
    news_link = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'news'


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
    k_remap = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'train'
