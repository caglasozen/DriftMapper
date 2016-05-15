package main.models.prior;

import main.models.distance.Distance;
import main.models.distance.TotalVariation;
import main.models.sampling.AbstractSampler;
import main.models.sampling.AllSamples;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by loongkuan on 11/05/16.
 **/
public class NaiveMatrix extends PriorModel{
    private int[] combinationFreq;

    public NaiveMatrix(Instances dataSet) {
        this.setDataSet(dataSet);
        this.reset();
    }

    public NaiveMatrix(AbstractSampler sampler) {
        this.setSampler(sampler);
        this.reset();
    }

    @Override
    public void reset() {
        int nCombinations = 1;
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        for (ArrayList<String> attributeValue : attributesValues) {
            nCombinations *= attributeValue.size();
        }
        combinationFreq = new int[nCombinations];
        for (Instance inst : this.sampler.getDataSet()) {
            double[] instValues = Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1);
            int hash = this.convertInstToHash(instValues);
            this.combinationFreq[hash] += 1;
        }
    }

    @Override
    public NaiveMatrix copy() {
        return new NaiveMatrix(this.sampler);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.sampler = new AllSamples(dataSet);
    }

    @Override
    public void setSampler(AbstractSampler sampler) {
        this.sampler = new AllSamples(sampler.getDataSet());
    }

    @Override
    public double findPv(double[] vector) {
        return this.vectorExists(vector) ?
                (double)this.combinationFreq[this.convertInstToHash(vector)]/(double)this.sampler.getDataSet().size() :
                0.0f;
    }

    private double[] max = new double[]{0.0, -1.0, -1.0};
    private double[] min = new double[]{1.0, -1.0, -1.0};
    private double mean = 0.0f;
    private double sd = 0.0f;
    @Override
    public double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample) {
        // Trim last attribute as allPossibleCombinations contains a class attribute wh
        Distance distanceMetric = new TotalVariation();

        double[] p = new double[sample.getNInstances()];
        double[] q = new double[sample.getNInstances()];
        double[] dist = new double[sample.getNInstances()];
        sample.reset();
        for (int i = 0; i < sample.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            p[i] = model1.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            q[i] = model2.findPv(Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            dist[i] = distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            if (dist[i] > max[0]) max = ArrayUtils.addAll(new double[]{dist[i]}, Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
            if (dist[i] < min[0]) min = ArrayUtils.addAll(new double[]{dist[i]}, Arrays.copyOfRange(inst.toDoubleArray(), 0, inst.numAttributes() - 1));
        }
        double finalDist = distanceMetric.findDistance(p, q);
        this.mean = 0.0f;
        for (int i = 0; i < sample.getNInstances(); i++) {
            mean += dist[i];
        }
        mean /= sample.getNInstances();
        this.sd = 0.0f;
        for (int i = 0; i < sample.getNInstances(); i++) {
            sd += Math.pow((dist[i] - mean), 2);
        }
        sd /= sample.getNInstances() - 1;
        sd = Math.sqrt(sd);
        return finalDist;
    }

    public double[] getMax(){
        return this.max;
    }

    public double[] getMin(){
        return this.min;
    }

    public double getMean() {
        return this.mean;
    }

    public double getSd() {
        return this.sd;
    }

    private boolean vectorExists(double[] vector) {
        ArrayList<ArrayList<String>> attributesValues = this.sampler.getAllPossibleValues();
        boolean exists = true;
        for (int attributeIndex = 0; attributeIndex < vector.length; attributeIndex++) {
            exists = exists && attributesValues.get(attributeIndex).contains(Double.toString(vector[attributeIndex]));
        }
        return exists;
    }

    private int convertInstToHash(double[] instance) {
        int hash = 0;
        for (int i = 0; i < instance.length; i++) {
            int current_hash = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_hash *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_hash *= this.sampler.getAllPossibleValues().get(i).indexOf(Double.toString(instance[i]));
            hash += current_hash;
        }
        return hash;
    }

    private double[] convertHashToInst(int hash) {
        double[] instance = new double[this.sampler.getAllPossibleValues().size()];
        for (int i = instance.length - 1; i >= 0; i--) {
            int current_value = 1;
            for (int j = i - 1; j >= 0; j--) {
                current_value *= this.sampler.getAllPossibleValues().get(j).size();
            }
            current_value *= (int) instance[i];
            instance[i] = hash % current_value;
            hash = hash / current_value;
        }
        return instance;
    }
}
