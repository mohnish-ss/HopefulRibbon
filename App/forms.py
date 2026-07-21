from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField
from wtforms.validators import DataRequired, Email, Optional, NumberRange


class ServiceForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[Optional(), Email()])
    postalcode = StringField('Postal Code', validators=[DataRequired()])
    
    # Breast cancer features
    radius = FloatField('Radius', validators=[DataRequired(), NumberRange(min=6.981, max=28.11)])
    texture = FloatField('Texture', validators=[DataRequired(), NumberRange(min=9.71, max=39.28)])
    perimeter = FloatField('Perimeter', validators=[DataRequired(), NumberRange(min=43.79, max=188.5)])
    
    submit = SubmitField('Submit')
