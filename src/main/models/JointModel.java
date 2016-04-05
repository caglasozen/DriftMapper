package main.models;

import main.models.posterior.PosteriorModel;
import main.models.prior.PriorModel;
import weka.core.Instances;

/**
 * Created by Lee on 30/01/2016.
 **/
public class JointModel extends AbstractModel{
    private PriorModel priorModel;
    private PosteriorModel posteriorModel;

    public JointModel(PriorModel priorModel, PosteriorModel posteriorModel) {
        this.priorModel = (PriorModel) priorModel.copy();
        this.posteriorModel = (PosteriorModel) posteriorModel.copy();
    }

    public JointModel(JointModel baseModel) {
        this.priorModel = (PriorModel) baseModel.priorModel.copy();
        this.posteriorModel = (PosteriorModel) baseModel.posteriorModel.copy();
    }

    public void setData(Instances data) {
        this.priorModel.setDataSet(data);
        this.posteriorModel.setData(data);
    }

    public static double pyGvModelDistance(JointModel model1, JointModel model2) {
        Instances domain = findIntersectionBetweenInstances(model1.posteriorModel.getAllPossibleInstances(),
                model2.posteriorModel.getAllPossibleInstances());
        return model1.posteriorModel.findDistance(model1.posteriorModel, model2.posteriorModel, domain);
    }

    public static double pvModelDistance(JointModel model1, JointModel model2){
        Instances domain = findIntersectionBetweenInstances(model1.priorModel.allPossibleInstances,
                model2.priorModel.allPossibleInstances);
        return model1.priorModel.findDistance(model1.priorModel, model2.priorModel, domain);
    }

    @Override
    public void reset() {}

    @Override
    public JointModel copy() {
        return new JointModel(this);
    }
}
