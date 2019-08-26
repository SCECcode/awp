def new_state(self, attr, init_value, new_value_fcn):
    """
    This function sets an attribute of an operator. If it is the first time
    the attribute is set, a copy of the operator is returned and the
    attribute is set to `init_value`. On the other hand, if the attribute
    already exists, then a reference to the operator is returned. The
    attribute is now updated using the function `new_value_fcn`.

    Arguments:
        attr: String specifying attribute to set.
        init_value : Initial value for attribute.
        new_value_fcn : The function to use to update the value if already
            exists. This function takes the one input argument, which is the
            old value of the attribute. 

    Returns:
        A copy if no value has been set. In this case, the field is set to
        `init_value`. Otherwise, returns itself.


    """
    if hasattr(self, attr):
        setattr(self, attr, new_value_fcn(getattr(self, attr)))
        return self
    else:
        obj = self.copy()
        setattr(obj, attr, init_value)
        return obj
