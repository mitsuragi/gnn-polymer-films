from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from sqlalchemy import ForeignKey

import datetime
from typing import Optional, List

class Base(DeclarativeBase):
    pass

class Parameter(Base):
    __tablename__ = 'Parameters'

    IdParameter: Mapped[int] = mapped_column(primary_key=True)
    ParameterCode: Mapped[Optional[str]]
    ParameterNameRu: Mapped[Optional[str]]
    ParameterNameEng: Mapped[Optional[str]]
    Symbol: Mapped[Optional[str]]
    IdParameterType: Mapped[Optional[int]] = mapped_column(ForeignKey('ParameterTypes.IdParameterType'))
    IdUnit: Mapped[Optional[int]] = mapped_column(ForeignKey('Units.IdUnit'))

    ParameterType: Mapped[Optional['ParameterType']] = relationship(back_populates='Parameters')
    Unit: Mapped[Optional['Unit']] = relationship(back_populates='Parameters')
    ParameterLimits: Mapped[List['Limit']] = relationship(back_populates='Parameter')

class ParameterType(Base):
    __tablename__ = 'ParameterTypes'

    IdParameterType: Mapped[int] = mapped_column(primary_key=True)
    Name: Mapped[Optional[str]]

    Parameters: Mapped[List['Parameter']] = relationship(back_populates='ParameterType')

# class ProductionLine(Base):
#     __tablename__ = 'ProductionLines'
#
#     IdLine: Mapped[int] = mapped_column(primary_key=True)
#     LineNumber: Mapped[str]
#     # DistanceBetweenExtruderFilmQualityMonitoringVideosystem: Mapped[float]
#     # DistanceBetweenCalenderThicknessGauge: Mapped[float]

class Unit(Base):
    __tablename__ = 'Units'

    IdUnit: Mapped[int] = mapped_column(primary_key=True)
    Sign: Mapped[Optional[str]]

    Parameters: Mapped[List['Parameter']] = relationship(back_populates='Unit')
    Coefficients: Mapped[List['Coefficient']] = relationship(back_populates='Unit')

class Coefficient(Base):
    __tablename__ = 'Coefficients'

    IdCoefficient: Mapped[int] = mapped_column(primary_key=True)
    CoefficientlName: Mapped[Optional[str]]
    IdUnit: Mapped[Optional[int]] = mapped_column(ForeignKey('Units.IdUnit'))
    Symbol: Mapped[Optional[str]]

    Unit: Mapped[Optional['Unit']] = relationship(back_populates='Coefficients')
    PolymersCoefficients: Mapped[List['PolymerCoefficient']] = relationship(back_populates='Coefficient')
    Polymers: Mapped[List['Polymer']] = relationship(secondary='PolymersCoefficients', viewonly=True)

class Polymer(Base):
    __tablename__ = 'Polymers'

    IdPolymerType: Mapped[int] = mapped_column(primary_key=True)
    PolymerType: Mapped[Optional[str]]
    Density: Mapped[Optional[float]]
    HeatCapacity: Mapped[Optional[float]]
    MeltingPoint: Mapped[Optional[float]]

    PolymerCoefficients: Mapped[List['PolymerCoefficient']] = relationship(back_populates='Polymer')
    Coefficients: Mapped[List['Coefficient']] = relationship(secondary='PolymersCoefficients', viewonly=True)
    Films: Mapped[List['Film']] = relationship(back_populates='Polymers')

class PolymerCoefficient(Base):
    __tablename__ = 'PolymersCoefficients'

    IdCoefficient: Mapped[int] = mapped_column(ForeignKey('Coefficients.IdCoefficient'), primary_key=True)
    IdPolymerType: Mapped[int] = mapped_column(ForeignKey('Polymers.IdPolymerType'), primary_key=True)
    CoefficientValue1: Mapped[Optional[float]]

    Polymer: Mapped[Optional['Polymer']] = relationship(back_populates='PolymerCoefficients')
    Coefficient: Mapped[Optional['Coefficient']] = relationship(back_populates='PolymersCoefficients')

class ParameterValue(Base):
    __tablename__ = 'ParameterValues'

    IdParameterValue: Mapped[int] = mapped_column(primary_key=True)
    DateTime: Mapped[Optional[datetime.datetime]]
    Value: Mapped[Optional[str]]
    IdParameter: Mapped[Optional[int]] = mapped_column(ForeignKey('Parameters.IdParameter'))

class Limit(Base):
    __tablename__ = 'Limits'

    IdLimit: Mapped[int] = mapped_column(primary_key=True)
    LowLimitValue: Mapped[Optional[float]]
    HighLimitValue: Mapped[Optional[float]]
    IdParameter: Mapped[Optional[int]] = mapped_column(ForeignKey('Parameters.IdParameter'))
    IdFilmType: Mapped[Optional[int]] = mapped_column(ForeignKey('Films.IdFilmType'))

    Parameter: Mapped[Optional['Parameter']] = relationship(back_populates='ParameterLimits')
    Film: Mapped[Optional['Film']] = relationship(back_populates='FilmLimits')

class Film(Base):
    __tablename__ = 'Films'

    IdFilmType: Mapped[int] = mapped_column(primary_key=True)
    FilmTypeCode: Mapped[Optional[str]]
    IdPolymerType: Mapped[Optional[int]] = mapped_column(ForeignKey('Polymers.IdPolymerType'))

    FilmLimits: Mapped[List['Limit']] = relationship(back_populates='Film')
    Polymers: Mapped[Optional['Polymer']] = relationship(back_populates='Films')

class NNCoefficientType(Base):
    __tablename__ = 'NNCoefficientTypes'

    IdType: Mapped[int] = mapped_column(primary_key=True)
    Name: Mapped[Optional[str]]

class NNCoefficient(Base):
    __tablename__ = 'NNCoefficients'

    IdCoefficient: Mapped[int] = mapped_column(primary_key=True)
    Value: Mapped[Optional[str]]
    IdCoefficientType: Mapped[Optional[int]] = mapped_column(ForeignKey('NNCoefficientTypes.IdType'))
