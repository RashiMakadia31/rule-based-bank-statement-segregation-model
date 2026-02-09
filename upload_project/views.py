from django.http import JsonResponse
from django.shortcuts import render

def custom_404(request, exception):
    if request.path.startswith('/api/'):
        return JsonResponse(
            {"error": "Endpoint not found"},
            status=404
        )
    return render(request, '404.html', status=404)
