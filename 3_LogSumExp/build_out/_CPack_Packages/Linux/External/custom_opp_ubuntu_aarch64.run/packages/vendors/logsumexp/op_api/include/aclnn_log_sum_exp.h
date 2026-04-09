
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LOG_SUM_EXP_H_
#define ACLNN_LOG_SUM_EXP_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLogSumExpGetWorkspaceSize
 * parameters :
 * x : required
 * dimOptional : optional
 * keepDim : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLogSumExpGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *dimOptional,
    bool keepDim,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLogSumExp
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLogSumExp(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
