from numpy import astype
import sqlalchemy as sa
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload
from torch import nn
from .models import *
import pandas as pd

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

def get_defect_limit(session: Session, id_defect: int):
    stmt = select(Limit.HighLimitValue).where(Limit.IdParameter == id_defect)

    res = session.execute(stmt).one_or_none()

    return res

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

    df_pivot = df_pivot.dropna(how='any')

    df_pivot = df_pivot.iloc[::step]

    df_pivot.reset_index(inplace=True)

    return df_pivot, stage_dict

def get_models(session: Session):
    stmt = select(NNModel.IdModel, NNModel.Name)

    models = session.execute(stmt).all()

    return models

def get_model_data(
    session: Session,
    model_id: int
):
    stmt = (
        select(NNModel)
        .where(NNModel.IdModel == model_id)
        .options(
            selectinload(NNModel.RelevantParameters),
            selectinload(NNModel.Coefficients)
            .selectinload(NNModelCoefficient.Coefficient)
            .selectinload(NNCoefficient.Type)
        )
    )

    model = session.scalar(stmt)

    if model is None:
        return None

    parameters = []
    defect = None

    for rel in model.RelevantParameters:
        if rel.IdParameterType == 1:
            parameters.append(rel.IdParameter)
        elif rel.IdParameterType == 2:
            defect = rel.IdParameter

    coefficients = {}
    
    for coef_rel in model.Coefficients:
        coef = coef_rel.Coefficient
        coef_type_name = coef_rel.Coefficient.Type.Name

        coefficients[coef_type_name] = coef.Value

    return {
        'model': model.Model,
        'parameters': parameters,
        'defect': defect,
        'coefficients': coefficients
    }

def get_forecasting_data(
    session: Session,
    parameters,
    window_length,
    step,
    from_datetime,
    to_datetime
):
    print(window_length)
    print(step)
    stmt_params = (
        select(Parameter.IdParameter, Parameter.ParameterCode)
        .where(Parameter.IdParameter.in_(parameters))
    )

    param_info = dict(session.execute(stmt_params).all())

    rows = []

    for param_id in parameters:
        limit = window_length * step

        stmt = (
            select(
                ParameterValue.IdParameter,
                ParameterValue.DateTime,
                ParameterValue.Value
            )
            .where(ParameterValue.IdParameter == param_id)
            .where(ParameterValue.DateTime >= from_datetime)
            .where(ParameterValue.DateTime <= to_datetime)
            .order_by(ParameterValue.DateTime.desc())
            .limit(limit)
        )

        values = session.execute(stmt).all()

        values = list(reversed(values))

        values = values[::step]

        rows.extend(values)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=['IdParameter', 'timestamp', 'value'])

    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    df_pivot = df.pivot(
        index='timestamp',
        columns='IdParameter',
        values='value'
    )

    new_columns = {
        param_id: param_info.get(param_id, f'param_{param_id}')
        for param_id in df_pivot.columns
    }

    df_pivot.rename(columns=new_columns, inplace=True)

    df_pivot.reset_index(inplace=True)

    return df_pivot

def save_model(
    session: Session,
    model_name,
    from_datetime,
    to_datetime,
    model,
    parameters,
    defect,
    coefficients
):
    nn_model = NNModel(
        Name=model_name,
        FromDateTime=from_datetime,
        ToDateTime=to_datetime,
        Model=model
    )

    session.add(nn_model)
    session.flush()

    relevant_params = []

    for param_id in parameters:
        relevant_params.append(
            NNModelRelevantParameter(
                IdModel=nn_model.IdModel,
                IdParameter=param_id,
                IdParameterType=1
            )
        )

    relevant_params.append(
        NNModelRelevantParameter(
            IdModel=nn_model.IdModel,
            IdParameter=defect,
            IdParameterType=2
        )
    )

    model_coeffs = []

    for coef_id in coefficients:
        model_coeffs.append(
            NNModelCoefficient(
                IdModel=nn_model.IdModel,
                IdCoefficient=coef_id
            )
        )

    session.add_all(relevant_params)
    session.add_all(model_coeffs)

    session.commit()

def delete_model(session: Session, model_id):
    model = session.get(NNModel, model_id)

    if model is None:
        return False

    # for rel in model.RelevantParameters:
    #     session.delete(rel)
    #
    # for coef in model.Coefficients:
    #     session.delete(coef)

    session.delete(model)

    session.commit()
