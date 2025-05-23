﻿using NeutralNET.Framework.Neural;
using NeutralNET.Matrices;

namespace NeutralNET.Validators;

public interface IValidator
{
    abstract void Validate(NeuralForward forward);
}
