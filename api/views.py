import os, uuid, math
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import PredictSerializer, AutoLearnSerializer, ApproveSerializer
from .services import (
    run_prediction_file,
    append_history,
    learn_account_mapping,
    promote_semantic_anchor,
    BankValidationWarning,
)
from .models import paths

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

class PredictAPIView(APIView):
    def post(self, request):
        s = PredictSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        v = s.validated_data
        
        upload_dir = paths(v["user_id"], v["bankid"])["uploads"]
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = f"{uuid.uuid4()}{os.path.splitext(v['file'].name)[1]}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb+") as f:
            for chunk in v["file"].chunks(): f.write(chunk)
            
        try:
            df = run_prediction_file(
                file_path,
                v["user_id"],
                v["bankid"],
                enforce_bank_validation=not v.get("bypass_bank_validation", False),
            )
            rows = sanitize_for_json(df.to_dict("records"))
            return Response({"upload_id": filename, "rows": rows})
        except BankValidationWarning as e:
            return Response(e.result, status=409)
        except ValueError as e:
            return Response({"error": str(e)}, status=400)
        except Exception as e:
            return Response({"error": "Internal server error"}, status=500)

class AutoLearnAPIView(APIView):
    def post(self, request):
        s = AutoLearnSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        v = s.validated_data
        
        append_history(v["user_id"], v["bankid"], v["particulars"], v["acname"])
        learn_account_mapping(v["user_id"], v["bankid"], v["particulars"], v["acname"])
        
        upload_dir = paths(v["user_id"], v["bankid"])["uploads"]
        file_path = os.path.join(upload_dir, v["upload_id"])
        
        df = run_prediction_file(
            file_path,
            v["user_id"],
            v["bankid"],
            enforce_bank_validation=False,
        )
        rows = sanitize_for_json(df.to_dict("records"))
        return Response({"rows": rows})

class ApproveAPIView(APIView):
    def post(self, request):
        s = ApproveSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        v = s.validated_data
        
        for row in v["rows"]:
            if row.get("acname"):
                append_history(v["user_id"], v["bankid"], row["Particulars"], row["acname"])
                promote_semantic_anchor(v["user_id"], v["bankid"], row["acname"])
        return Response({"message": "Saved successfully"})
