package main.models.frequency;

import main.experiments.ExperimentResult;
import main.models.Model;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Set;

/**
 * Created by loongkuan on 28/11/2016.
 **/

public abstract class BaseFrequencyModel extends Model {
    protected int totalFrequency;
    // TODO: get rid of att avail
    protected int[] attributesAvailable;
    protected int nAttributesActive;
    protected Instance exampleInst;
    protected Instances allInstances;
    protected int[] hashBases;

    public abstract void changeAttributeSubsetLength(int length);

    protected abstract int findFv(Instance instance, int[] attributesSubset);
    protected abstract int findFy(int classIndex);
    protected abstract int findFvy(Instance instance, int[] attributesSubset, int classIndex);
    protected abstract Set<Integer> getAllHashes(int[] attributeSubset);

    protected static int instanceToPartialHash(Instance instance, int[] activeAttributes, int[] hashBases) {
        int hash = 0;
        // TODO: Don't rely on class being last attribute
        // Iterate over attributes in strippedInstance
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            // If not class or active attribute set missing
            if (ArrayUtils.contains(activeAttributes, i)) {
                hash += hashBases[i] * (int)instance.value(i);
            }
        }
        return hash;
    }

    protected static Instance partialHashToInstance(int partialHash, int[] activeAttributes, Instance exampleInst, int[] hashBases) {
        Instance partialInstance = new DenseInstance(exampleInst.numAttributes());
        partialInstance.setDataset(exampleInst.dataset());

        for (int i = partialInstance.numAttributes() - 2; i >= 0; i--) {
            if (ArrayUtils.contains(activeAttributes, i)) {
                partialInstance.setValue(i, partialHash / hashBases[i]);
                partialHash = partialHash % hashBases[i];
            }
            else {
                partialInstance.setMissing(i);
            }
        }
        partialInstance.setClassMissing();
        return partialInstance;
    }

    protected static boolean isPartialHashEqualHash(int partialHash, int hash, int[] attributeSubset,
                                                    int[] hashBases, Instance exampleInst) {
        int rem = hash - partialHash;
        int quo;
        for (int i = exampleInst.numAttributes() - 2; i >= 0; i--) {
            if (rem < 0) return false;
            if (!ArrayUtils.contains(attributeSubset, i)) {
                quo = rem / hashBases[i];
                if (quo >= exampleInst.attribute(i).numValues()) return false;
                rem = rem % hashBases[i];
            }
        }
        return rem == 0;
    }

    protected boolean validAttributeSubset(int[] attributeSubset) {
        boolean valid = true;
        for (int attributeIndex : attributeSubset) {
            valid = valid && ArrayUtils.contains(this.attributesAvailable, attributeIndex);
        }
        return valid && attributeSubset.length == this.nAttributesActive;
    }

    protected static ArrayList<Integer> sampleHashes(ArrayList<Integer> hashes, double sampleScale) {
        ArrayList<Integer> sampledHashes = new ArrayList<>();
        ArrayList<Integer> allHash = new ArrayList<>(hashes);
        Collections.shuffle(allHash);
        for (int i = 0; i < Math.min((int)(hashes.size() / sampleScale), hashes.size()); i++) {
            sampledHashes.add(allHash.get(i));
        }
        return sampledHashes;
    }

    private ArrayList<Integer> mergeHashes(BaseFrequencyModel model, int[] attributeSubset) {
        Set<Integer> hashes = this.getAllHashes(attributeSubset);
        hashes.addAll(model.getAllHashes(attributeSubset));
        return new ArrayList<>(hashes);
    }

    // TODO: Try and separate these into smaller functions
    @Override
    public ArrayList<ExperimentResult> findCovariateDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);
        allHashes = sampleHashes(allHashes, sampleScale);

        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        double[] separateDistance = new double[allHashes.size()];
        double[][] instanceValues = new double[allHashes.size()][this.exampleInst.numAttributes()];
        //TODO: Remove debug flags
        int totalpSeen = 0;
        int totalqSeen = 0;
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = partialHashToInstance(allHashes.get(i), attributeSubset, this.exampleInst, hashBases);
            totalpSeen += this.findFv(instance, attributeSubset);
            totalqSeen += modelAfter.findFv(instance, attributeSubset);
            p[i] = this.totalFrequency == 0 ?
                    0 : (double)this.findFv(instance, attributeSubset) / (double)this.totalFrequency;
            q[i] = modelAfter.totalFrequency == 0 ?
                    0 : (double)modelAfter.findFv(instance, attributeSubset) / (double)modelAfter.totalFrequency;
            separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
            instanceValues[i] = instance.toDoubleArray();
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        returnResults.add(finalResult);
        return returnResults;
    }

    @Override
    public ArrayList<ExperimentResult> findJointDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        int nClass =  exampleInst.numClasses();
        double[] p = new double[allHashes.size() * nClass];
        double[] q = new double[allHashes.size() * nClass];
        double[] separateDistance = new double[allHashes.size() * nClass];
        double[][] instanceValues = new double[allHashes.size() * nClass][this.exampleInst.numAttributes()];
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = partialHashToInstance(allHashes.get(i), attributeSubset, this.exampleInst, hashBases);
            for(int classIndex = 0; classIndex < nClass; classIndex++) {
                p[i*nClass + classIndex] = this.totalFrequency == 0 ?
                        0 : (double)this.findFvy(instance, attributeSubset, classIndex) / (double)this.totalFrequency;
                q[i*nClass + classIndex] = modelAfter.totalFrequency == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double)modelAfter.totalFrequency;
                separateDistance[i*nClass + classIndex] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
                instanceValues[i*nClass + classIndex] = instance.toDoubleArray();
            }
        }
        double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
        ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
        returnResults.add(finalResult);
        return returnResults;
    }

    @Override
    public ArrayList<ExperimentResult> findLikelihoodDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[allHashes.size()];
        double[] q = new double[allHashes.size()];
        double[] separateDistance = new double[allHashes.size()];
        double[][] instanceValues = new double[allHashes.size()][this.exampleInst.numAttributes()];
        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        for (int classIndex = 0; classIndex < this.exampleInst.numClasses(); classIndex++) {
            for (int i = 0; i < allHashes.size(); i++) {
                Instance instance = partialHashToInstance(allHashes.get(i), attributeSubset, this.exampleInst, hashBases);
                p[i] = this.findFy(classIndex) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFy(classIndex);
                q[i] = modelAfter.findFy(classIndex) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFy(classIndex);
                separateDistance[i] = this.distanceMetric.findDistance(new double[]{p[i]}, new double[]{q[i]});
                instanceValues[i] = instance.toDoubleArray();
            }
            double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
            ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
            returnResults.add(finalResult);
        }
        return returnResults;
    }

    @Override
    public ArrayList<ExperimentResult> findPosteriorDistance(Model modelToCompare, int[] attributeSubset, double sampleScale) {
        if (!validAttributeSubset(attributeSubset)) return null;
        BaseFrequencyModel modelAfter = (BaseFrequencyModel)modelToCompare;

        // Get the hash of all the seen instances and get a sample from them
        ArrayList<Integer> allHashes = mergeHashes(modelAfter, attributeSubset);

        double[] p = new double[this.exampleInst.numClasses()];
        double[] q = new double[this.exampleInst.numClasses()];
        double[] separateDistance = new double[this.exampleInst.numClasses()];
        double[][] instanceValues = new double[this.exampleInst.numClasses()][this.exampleInst.numAttributes()];
        ArrayList<ExperimentResult> returnResults = new ArrayList<>();
        for (int i = 0; i < allHashes.size(); i++) {
            Instance instance = partialHashToInstance(allHashes.get(i), attributeSubset, this.exampleInst, hashBases);
            for (int classIndex = 0; classIndex < this.exampleInst.numClasses(); classIndex++) {
                instance.setClassValue(classIndex);
                p[classIndex] = this.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) this.findFvy(instance, attributeSubset, classIndex) / (double) this.findFv(instance, attributeSubset);
                q[classIndex] = modelAfter.findFv(instance, attributeSubset) == 0 ?
                        0 : (double) modelAfter.findFvy(instance, attributeSubset, classIndex) / (double) modelAfter.findFv(instance, attributeSubset);
                separateDistance[classIndex] = this.distanceMetric.findDistance(new double[]{p[classIndex]}, new double[]{q[classIndex]});
                instanceValues[classIndex] = instance.toDoubleArray();
            }
            double finalDistance = this.distanceMetric.findDistance(p, q) * sampleScale;
            ExperimentResult finalResult = new ExperimentResult(finalDistance, separateDistance, instanceValues);
            returnResults.add(finalResult);
        }
        return returnResults;
    }
}
