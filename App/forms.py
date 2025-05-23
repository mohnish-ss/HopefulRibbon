from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField
from wtforms.validators import DataRequired, Email, Optional


class ServiceForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    postalcode = StringField('Postal Code', validators=[DataRequired()])
    
    # Breast cancer features
    radius = FloatField('Radius', validators=[DataRequired()])
    texture = FloatField('Texture', validators=[DataRequired()])
    perimeter = FloatField('Perimeter', validators=[DataRequired()])
    
    submit = SubmitField('Submit')
