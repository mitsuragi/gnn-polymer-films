import sqlalchemy as sa
from sqlalchemy import func, select
from .models import *

def create_engine(url: str) -> sa.Engine:
    return sa.create_engine(url, echo=True)

def get_parameters(engine: sa.Engine):
    with engine.connect() as conn:
        stmt = select(Parameter.IdParameter, Parameter.ParameterNameRu).where(Parameter.IdParameterType == 3)

        res = conn.execute(stmt)

        return res.all()

def get_defects(engine: sa.Engine):
    with engine.connect() as conn:
        stmt = select(Parameter.ParameterNameRu, Parameter.IdParameter).where(Parameter.IdParameterType == 2)

        res = conn.execute(stmt)

        return res.all()

def get_datetime_range(engine: sa.Engine):
    with engine.connect() as conn:
        stmt = select(func.min(ParameterValue.DateTime), func.max(ParameterValue.DateTime)).where(ParameterValue.IdParameter != 80)

        min_dt, max_dt = conn.execute(stmt).one()

        return min_dt, max_dt

def get_nn_coeffs(engine: sa.Engine):
    with engine.connect() as conn:
        stmt = select(NNCoefficient)

        res = conn.execute(stmt).all()

        return res
