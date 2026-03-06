import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from .models import *
import pandas as pd

def create_engine(url: str) -> sa.Engine:
    return sa.create_engine(url, echo=True)

def get_parameters(session: Session):
    stmt = select(Parameter.IdParameter, Parameter.ParameterNameRu).where(Parameter.IdParameterType == 3)

    res = session.execute(stmt)

    return res.all()

def get_defects(session: Session):
    stmt = select(Parameter.ParameterNameRu, Parameter.IdParameter).where(Parameter.IdParameterType == 2)

    res = session.execute(stmt)

    return res.all()

def get_datetime_range(session: Session):
    stmt = select(func.min(ParameterValue.DateTime), func.max(ParameterValue.DateTime)).where(ParameterValue.IdParameter != 80)

    min_dt, max_dt = session.execute(stmt).one()

    return min_dt, max_dt

def get_nn_coeffs(session: Session):
    stmt = select(NNCoefficient)

    res = session.scalars(stmt)

    return res.all()

def get_training_data(
    session: Session,
    defect_id: int,
    parameter_ids: list[int],
    step: int,
    time_from: datetime.datetime,
    time_to: datetime.datetime
):
    all_ids = parameter_ids + [defect_id]

    stmt_params = select(Parameter.IdParameter, Parameter.ParameterCode).where(Parameter.IdParameter.in_(all_ids))

    stmt_stage = (
        select(
            Stage.Name,
            Parameter.ParameterCode
        )
        .join(Parameter, Parameter.IdStage == Stage.IdStage)
        .where(Parameter.IdParameter.in_(parameter_ids))
    )

    stage_rows = session.execute(stmt_stage).all()

    print('stage_rows: ', stage_rows)

    stage_dict = {}

    for stage_name, param_code in stage_rows:
        if stage_name not in stage_dict:
            stage_dict[stage_name] = []

        stage_dict[stage_name].append(param_code)

    param_info = dict(session.execute(stmt_params).all())

    stmt_values = (
        select(
            ParameterValue.IdParameter,
            ParameterValue.DateTime,
            ParameterValue.Value
        )
        .where(ParameterValue.IdParameter.in_(all_ids))
        .where(ParameterValue.DateTime >= time_from)
        .where(ParameterValue.DateTime <= time_to)
        .order_by(ParameterValue.DateTime)
    )

    rows = session.execute(stmt_values).all()

    df = pd.DataFrame(rows, columns=['IdParameter', 'timestamp', 'value'])

    if df.empty:
        return pd.DataFrame()

    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df = df.drop_duplicates(subset=['timestamp', 'IdParameter'])

    df_pivot = df.pivot(
        index='timestamp',
        columns='IdParameter',
        values='value'
    )

    new_columns = {}

    for param_id in df_pivot.columns:
        if param_id == defect_id:
            new_columns[param_id] = 'target'
        else:
            new_columns[param_id] = param_info.get(param_id, f'param_{param_id}')

    df_pivot.rename(columns=new_columns, inplace=True)

    print(df_pivot.head())
    df_pivot = df_pivot.dropna(how='any')

    df_pivot = df_pivot.iloc[::step]

    df_pivot.reset_index(inplace=True)

    return df_pivot, stage_dict
