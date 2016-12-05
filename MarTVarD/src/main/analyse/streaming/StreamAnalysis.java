package main.analyse.streaming;

import main.models.Model;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 2/12/2016.
 */
public abstract class StreamAnalysis {

    public abstract void reset();
    public abstract void addInstance(Instance instance);

    protected Model baseModel;
    protected Model movingModel;
}
