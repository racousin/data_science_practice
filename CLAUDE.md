# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## PRIMARY FOCUS: Data Science Practice Course Review & Improvement

We are reviewing and improving the Data Science Practice course week by week. The course content is located in `website/public/repositories/data_science_practice` and uses a dedicated GitHub repository for student submissions.

### Course Structure

- **Repository**: `data_science_practice_2025` (student submission repository)
- **Website Integration**: Students update their answers directly to the repository with validation through the website
- **Content Location**: `/website/public/repositories/data_science_practice/`
- **Workflows**: `.github/workflows/` contains automation for validation and permissions

### Development Guidelines

1. **Tone**: Maintain neutral and scientific tone throughout all content
2. **Weekly Review Process**:
   - Review existing content for clarity and accuracy
   - Add exercises where needed to reinforce concepts
   - Implement tests to validate student submissions
   - Ensure workflows are functioning correctly
3. **Testing**: Regularly test GitHub workflows to ensure validation pipeline works
4. **Exercise Development**:
   - Each exercise should have clear objectives
   - Include automated tests where applicable
   - Ensure compatibility with the website validation system
5. **Student Workflow**:
   - Students push to `data_science_practice_2025` repository
   - Automatic validation through website integration
   - Immediate feedback on submissions

### Technical Requirements

1. **Consistency**: All exercises must follow established patterns
2. **Progressive Difficulty**: Build complexity gradually throughout the course
3. **Automated Validation**: All exercises should include validation tests
4. **Clear Instructions**: Every task should have unambiguous requirements
5. **Website Integration**: Ensure all content works with the validation system

### Workflow Testing Checklist

- [ ] Verify GitHub Actions are triggered correctly
- [ ] Test student submission validation
- [ ] Check website integration endpoints
- [ ] Validate permission enforcement
- [ ] Test PR review tracking