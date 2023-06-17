from django import forms

class DiseasePredictionForm(forms.Form):
    symptoms = forms.CharField(widget=forms.Textarea)
