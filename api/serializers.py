from rest_framework import serializers

class PredictSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    bankid = serializers.CharField()
    file = serializers.FileField()

class HistoryRowSerializer(serializers.Serializer):
    Particulars = serializers.CharField()
    acname = serializers.CharField(required=False, allow_blank=True, allow_null=True)

class ApproveSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    bankid = serializers.CharField()
    rows = serializers.ListField(child=HistoryRowSerializer())

class AutoLearnSerializer(serializers.Serializer):
    user_id = serializers.CharField()
    bankid = serializers.CharField()
    upload_id = serializers.CharField()
    particulars = serializers.CharField()
    acname = serializers.CharField(required=False, allow_blank=True, allow_null=True)
