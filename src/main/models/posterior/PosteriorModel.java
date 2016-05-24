package main.models.posterior;

import main.models.sampling.AbstractSampler;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by loongkuan on 1/04/16.
 **/
public interface PosteriorModel {
    void setDataSet(Instances data);
    void setSampler(AbstractSampler sampler);
    double findPyGv(double classValue, Instance vector);
    double findDistance(PosteriorModel model1, PosteriorModel model2, Instances domain);
    PosteriorModel copy();
}
