�	 ����2@ ����2@! ����2@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails- ����2@Wv����1@1� L���?A�e��a��?I��Ӝ���?*	�����LH@2U
Iterator::Model::ParallelMapV2%u��?!Ix�5?>@)%u��?1Ix�5?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!�^B{	�9@)/�$��?1X�<ݚ5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�HP��?!�_cV9@)	�^)ˀ?1hj�pW�0@:Preprocessing2F
Iterator::Model/�$��?!X�<ݚE@) �o_�y?1�^B{	�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!����u @)����Mbp?1����u @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Ziplxz�,C�?!�[��"eL@)Ǻ���f?1��s��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�����H@)�J�4a?1�����H@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIA�ʋ�X@Q��/O�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Wv����1@Wv����1@!Wv����1@      ��!       "	� L���?� L���?!� L���?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	��Ӝ���?��Ӝ���?!��Ӝ���?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qA�ʋ�X@y��/O�?�"5
sequential/dense/MatMulMatMul�r�g�?!�r�g�?0"C
%gradient_tape/sequential/dense/MatMulMatMul$��ۦ[�?!�OIa�?0"7
sequential/dense_1/MatMulMatMulI��Lt�?!ԧDn3��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�){��r�?!�����?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�){��r�?!Pr��[�?"!
Adam/PowPow�t�,]�?!��%}g�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam[L�$�D�?!k�9���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam[L�$�D�?!���n���?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam[L�$�D�?!���	� �?"[
AArithmeticOptimizer/ReorderCastLikeAndValuePreserving_double_CastCast[L�$�D�?!����D�?Q      Y@Y�M�_{4@a��(�S@q{��vX@y}w�*��?"�
both�Your program is POTENTIALLY input-bound because 92.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�5.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 