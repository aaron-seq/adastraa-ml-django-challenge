from django.db import models


class Prediction(models.Model):
    """
    Store prediction history.
    """
    uploaded_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    result_file = models.FileField(upload_to='results/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    num_predictions = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Prediction {self.id} - {self.created_at}"
