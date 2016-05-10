package main.models.prior;

import explorer.ChordalysisModelling;
import main.generator.componets.BayesianNetworkGenerator;
import main.models.distance.Distance;
import main.models.distance.HellingerDistance;
import main.models.distance.TotalVariation;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 1/04/16.
 **/
public class BayesianNetwork extends PriorModel{
    private BayesianNetworkGenerator bnModel;

    public BayesianNetwork(Instances dataSet) {
        this.sampler = new AllSamples(dataSet);
        reset();
    }

    public BayesianNetwork(AbstractSampler sampler) {
        this.sampler = sampler;
        reset();
    }

    public void setDataSet(Instances dataSet) {
        this.sampler.setDataSet(dataSet);
        reset();
    }

    public void setSampler(AbstractSampler sampler) {
        this.sampler = sampler;
        reset();
    }

    private void generateBayesNet() {
        Instances dataSet = sampler.getDataSet();
        Instances trimmedInstances = trimClass(dataSet);

        String[] variablesNames = new String[dataSet.numAttributes()];
        for (int i = 0; i < variablesNames.length; i++) {
            variablesNames[i] = dataSet.attribute(i).name();
        }

        // Chordalysis modeler with 0.1G of memory allocated
        ChordalysisModelling modeller = new ChordalysisModelling(0.05);
        modeller.buildModel(new Instances(trimmedInstances));

        bnModel = new BayesianNetworkGenerator(modeller, variablesNames, sampler.getAllPossibleValues());
    }

    @Override
    public BayesianNetwork copy() {
        return new BayesianNetwork(this.sampler);
    }

    @Override
    public void reset() {
        generateBayesNet();
    }

    @Override
    public double findPv(double[] vector) {
        try {
            int[] dataVector = new int[vector.length];
            for (int i = 0; i < vector.length; i++) {
                dataVector[i] = this.bnModel.nodes[i].getOutcomeIndex(Double.toString(vector[i]));
            }
            return bnModel.getJointProbabilityOfX(dataVector);
        }
        catch (IllegalArgumentException ex) {
            System.out.print("e ");
            return 0.0f;
        }
    }

    @Override
    public double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample) {
        // Trim last attribute as allPossibleCombinations contains a class attribute wh
        Distance distanceMetric = new TotalVariation();

        double[] p = new double[sample.getNInstances()];
        double[] q = new double[sample.getNInstances()];
        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            p[i] = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            q[i] = model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
        }
        return distanceMetric.findDistance(p, q);
    }
}
